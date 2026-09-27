"""Host-neutral Exec facts and conservative shared risk matching."""

from __future__ import annotations

import ntpath
import os
import posixpath
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

type ExecShellFamily = Literal["bash", "powershell", "pwsh"]
type ExecShellSelector = Literal["auto", "powershell", "pwsh"]
type ExecPlatform = Literal["posix", "windows"]
type ExecSyntaxConfidence = Literal["high", "medium", "low", "unknown"]
type ExecInspectorStatus = Literal["available", "uncertain", "failed", "timeout"]
type ExecPathRole = Literal["read", "write"]
type ExecIdentityKind = Literal[
    "builtin",
    "cmdlet",
    "native",
    "alias",
    "function",
    "script",
    "shim",
    "workspace",
    "ambiguous",
    "unknown",
]
EXEC_CONFIRMATION_REASON: Final = (
    "Exec inspection was unavailable or uncertain and requires confirmation."
)
EXEC_CATASTROPHIC_REASON: Final = (
    "The Exec command matches a known catastrophic operation and requires confirmation."
)
EXEC_DESTRUCTIVE_REASON: Final = (
    "The Exec command matches a known destructive operation and requires confirmation."
)


@dataclass(frozen=True, slots=True)
class ResolvedExecShell:
    """The immutable process-lifetime shell selection shared by inspection and execution."""

    selector: ExecShellSelector
    platform: ExecPlatform
    family: ExecShellFamily
    executable: str | None
    flags: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    available: bool
    diagnostic: str | None = None
    version: tuple[int, ...] | None = None

    @property
    def canonical_executable(self) -> str | None:
        """Return the selected executable identity without exposing PATH contents."""
        return self.executable

    @property
    def env(self) -> dict[str, str]:
        """Return a detached environment mapping for one host process."""
        return dict(self.environment)

    @property
    def shell(self) -> ExecShellFamily:
        """Return the canonical host family."""
        return self.family


@dataclass(frozen=True, slots=True)
class ExecCommandIdentity:
    """A command identity observed by a host inspector."""

    requested: str
    resolved: str | None = None
    kind: ExecIdentityKind = "unknown"
    canonical: str | None = None
    module: str | None = None
    resolution_count: int | None = None


@dataclass(frozen=True, slots=True)
class ExecPathAccess:
    """A path operand and its coarse read/write role."""

    path: str
    role: ExecPathRole


@dataclass(frozen=True, slots=True)
class ExecGrammarClassification:
    """Code-owned grammar facts used by the strict low-permission policy."""

    accepted: bool
    reason: str
    file_accesses: tuple[ExecPathAccess, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecDynamicConstruct:
    """A syntax construct whose target or effects are not statically fixed."""

    kind: str
    expression: str


@dataclass(frozen=True, slots=True)
class CatastrophicMatch:
    """One fixed matcher hit; policy mapping belongs outside the Host."""

    rule: str
    evidence: str


@dataclass(frozen=True, slots=True)
class ExecAssessment:
    """Shared inspection vocabulary returned by every Exec Host."""

    syntax_confidence: ExecSyntaxConfidence
    syntax_uncertain: bool
    command_identities: tuple[ExecCommandIdentity, ...] = ()
    file_accesses: tuple[ExecPathAccess, ...] = ()
    network_targets: tuple[str, ...] = ()
    dynamic_constructs: tuple[ExecDynamicConstruct, ...] = ()
    catastrophic_matches: tuple[CatastrophicMatch, ...] = ()
    destructive_matches: tuple[str, ...] = ()
    git_delegation_safe: bool | None = None
    inspector_status: ExecInspectorStatus = "available"
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_identities", tuple(self.command_identities))
        object.__setattr__(self, "file_accesses", tuple(self.file_accesses))
        object.__setattr__(self, "network_targets", tuple(self.network_targets))
        object.__setattr__(self, "dynamic_constructs", tuple(self.dynamic_constructs))
        object.__setattr__(self, "catastrophic_matches", tuple(self.catastrophic_matches))
        object.__setattr__(self, "destructive_matches", tuple(self.destructive_matches))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def uncertain(self) -> bool:
        """Return whether inspection failed closed to an unknown classification."""
        return self.syntax_uncertain or self.inspector_status != "available"

    @classmethod
    def uncertain_result(
        cls,
        reason: str,
        *,
        status: ExecInspectorStatus = "uncertain",
        catastrophic: tuple[CatastrophicMatch, ...] = (),
    ) -> ExecAssessment:
        return cls(
            syntax_confidence="unknown",
            syntax_uncertain=True,
            catastrophic_matches=catastrophic,
            inspector_status=status,
            diagnostics=(reason,),
        )


@dataclass(frozen=True, slots=True)
class ExecOutcome:
    """Raw host process output. It intentionally has no Tool-layer fields."""

    exit_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False

    @property
    def returncode(self) -> int | None:
        """Expose subprocess terminology without changing the raw boundary."""
        return self.exit_code


_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"https?://[^\s\"'`<>]+",
    re.IGNORECASE,
)
_ASSIGNMENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[;|&\n])\s*[A-Za-z_][A-Za-z0-9_]*\s*=",
)
_SUBSTITUTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"\$\(|`[^`]*`|<\([^)]*\)", re.DOTALL)
_VARIABLE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})")
_REDIRECTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?:^|\s)(?:\d*>>?|\d*<)\s*[^\s]+")
_POWERSHELL_SCRIPTBLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\{.*\}",
    re.DOTALL,
)


def catastrophic_matches(command: str) -> tuple[CatastrophicMatch, ...]:
    """Return invocation-aware catastrophic facts without deciding a permission level."""
    if not isinstance(command, str):
        raise TypeError("Exec command must be a string")
    matches: list[CatastrophicMatch] = []

    def add(rule: str, evidence: str) -> None:
        if all(existing.rule != rule for existing in matches):
            matches.append(CatastrophicMatch(rule=rule, evidence=evidence))

    for segment in _command_segments(command):
        invocation = _unwrap_invocation(segment)
        if not invocation:
            continue
        name = _command_basename(invocation[0])
        evidence = " ".join(invocation)

        deletion = _deletion_facts(invocation)
        if deletion is not None:
            recursive, force, targets = deletion
            if recursive and force and any(_is_broad_target(target) for target in targets):
                add("broad-recursive-force-delete", evidence)
            if recursive and force and any(_is_dynamic_target(target) for target in targets):
                add("dynamic-recursive-force-delete", evidence)

        if name in {
            "clear-content",
            "clc",
            "set-content",
            "sc",
            "add-content",
            "ac",
            "out-file",
        }:
            content_targets = _path_option_targets(invocation[1:])
            if any(_is_broad_target(target) for target in content_targets):
                add("broad-content-clear", evidence)

        if name in {
            "diskpart",
            "format",
            "initialize-disk",
            "format-volume",
            "clear-disk",
            "wipefs",
        } or name.startswith("mkfs."):
            add("disk-initialization-or-format", evidence)
        if name in {"remove-volume", "remove-partition", "remove-disk"}:
            add("volume-or-partition-removal", evidence)
        if name == "delete" and len(invocation) > 1 and invocation[1].lower() in {
            "volume",
            "partition",
        }:
            add("volume-or-partition-removal", evidence)

        if _overwrites_device(invocation):
            add("device-overwrite", evidence)

        git = _git_invocation(invocation)
        if git is not None:
            subcommand, arguments = git
            if subcommand == "clean" and _git_clean_is_catastrophic(arguments):
                add("git-clean-all", evidence)
            elif subcommand == "reset" and any(
                argument.lower() == "--hard" for argument in arguments
            ):
                add("git-reset-hard", evidence)
            elif subcommand == "checkout" and _git_checkout_is_broad_force(arguments):
                add("git-force-checkout-or-restore", evidence)
            elif subcommand == "restore" and _git_restore_is_broad(arguments):
                add("git-force-checkout-or-restore", evidence)
            alias_command = _git_alias_command(invocation, subcommand)
            if alias_command is not None:
                for nested_match in catastrophic_matches(alias_command):
                    add(nested_match.rule, nested_match.evidence)

        if name in {
            "shutdown",
            "reboot",
            "poweroff",
            "halt",
            "stop-computer",
            "restart-computer",
        }:
            add("shutdown-or-reboot", evidence)
        if name in {"systemctl", "loginctl"} and _control_command_verb(
            invocation[1:]
        ) in {"reboot", "poweroff", "halt"}:
            add("shutdown-or-reboot", evidence)

        if name in {"eval", "exec"} and len(invocation) > 1:
            for nested_match in catastrophic_matches(" ".join(invocation[1:])):
                add(nested_match.rule, nested_match.evidence)

        nested = _nested_shell_command(invocation)
        if nested is not None:
            for nested_match in catastrophic_matches(nested):
                add(nested_match.rule, nested_match.evidence)

    unquoted = _unquoted_source(command)
    fork_bomb = re.search(
        r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:",
        unquoted,
    )
    if fork_bomb is not None:
        add("fork-bomb", fork_bomb.group(0))
    return tuple(matches)


def bash_recursive_forced_delete_targets(command: str) -> tuple[str, ...]:
    """Return statically tokenized targets from recursive forced Bash rm calls."""
    targets: list[str] = []
    for segment in _command_segments(command):
        invocation = _unwrap_invocation(segment)
        if not invocation or _command_basename(invocation[0]) != "rm":
            continue
        deletion = _deletion_facts(invocation)
        if deletion is None:
            continue
        recursive, force, invocation_targets = deletion
        if recursive and force:
            targets.extend(invocation_targets)
    return tuple(targets)


def destructive_matches(command: str) -> tuple[str, ...]:
    """Return typed facts for known destructive command forms."""
    if not isinstance(command, str):
        raise TypeError("Exec command must be a string")
    matches: list[str] = []

    def add(rule: str) -> None:
        if rule not in matches:
            matches.append(rule)

    for segment in _command_segments(command):
        invocation = _unwrap_invocation(segment)
        if not invocation:
            continue
        name = _command_basename(invocation[0])
        deletion = _deletion_facts(invocation)
        if deletion is not None:
            recursive, force, _targets = deletion
            if recursive or force:
                add("recursive-or-forced-delete")
        lowered = tuple(token.lower() for token in invocation[1:])
        if name in {"del", "erase"} and any(
            "f" in token[1:] for token in lowered if token.startswith("/")
        ):
            add("forced-windows-delete")
        if name == "dd" and any(token.startswith("if=") for token in lowered):
            add("device-read-or-write")
    return tuple(matches)


def _command_segments(command: str) -> tuple[tuple[str, ...], ...]:
    segments: list[tuple[str, ...]] = []
    tokens: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    index = 0

    def flush_token() -> None:
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    def flush_segment() -> None:
        flush_token()
        if tokens:
            segments.append(tuple(tokens))
            tokens.clear()

    while index < len(command):
        character = command[index]
        if quote is not None:
            if character == quote:
                quote = None
            elif character in {"`", "\\"} and index + 1 < len(command):
                next_character = command[index + 1]
                if character == "`" or next_character in {quote, "$", "`"}:
                    buffer.append(next_character)
                    index += 1
                else:
                    buffer.append(character)
            else:
                buffer.append(character)
        elif character in {"'", '"'}:
            quote = character
        elif character == "#" and not buffer:
            while index < len(command) and command[index] != "\n":
                index += 1
            flush_segment()
        elif character.isspace():
            flush_token()
            if character == "\n":
                flush_segment()
        elif character in ";|&":
            flush_segment()
        elif character in "<>":
            flush_token()
            operator = character
            if index + 1 < len(command) and command[index + 1] == character:
                operator += character
                index += 1
            tokens.append(operator)
        elif character == "\\" and index + 1 < len(command):
            next_character = command[index + 1]
            if next_character.isspace() and re.fullmatch(r"[A-Za-z]:", "".join(buffer)):
                buffer.append(character)
            elif next_character in {"'", '"'}:
                buffer.append(next_character)
                index += 1
            else:
                buffer.append(character)
        elif character == "`" and index + 1 < len(command):
            buffer.append(command[index + 1])
            index += 1
        else:
            buffer.append(character)
        index += 1
    flush_segment()
    return tuple(segments)


def _unwrap_invocation(tokens: tuple[str, ...]) -> tuple[str, ...]:
    values = list(tokens)
    while values and values[0] in {"(", ")", "{", "}", "."}:
        values.pop(0)
    if values:
        values[0] = values[0].lstrip("({")
        if not values[0]:
            values.pop(0)
    while values and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", values[0]):
        values.pop(0)
    if not values:
        return ()
    name = _command_basename(values[0])
    if name in {"sudo", "doas"}:
        values.pop(0)
        _consume_privilege_wrapper_options(values)
    elif name == "env":
        values.pop(0)
        while values and (
            values[0].startswith("-")
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", values[0]) is not None
        ):
            values.pop(0)
    elif name in {"command", "builtin", "nohup"}:
        values.pop(0)
        while values and values[0].startswith("-"):
            values.pop(0)
    return tuple(values)


_PRIVILEGE_VALUE_OPTIONS: Final[frozenset[str]] = frozenset(
    {
        "--chdir",
        "--chroot",
        "--close-from",
        "--group",
        "--host",
        "--prompt",
        "--role",
        "--type",
        "--user",
        "-C",
        "-D",
        "-g",
        "-h",
        "-p",
        "-r",
        "-t",
        "-u",
    }
)


def _consume_privilege_wrapper_options(values: list[str]) -> None:
    while values:
        argument = values[0]
        if argument == "--":
            values.pop(0)
            return
        name = argument.split("=", maxsplit=1)[0]
        if name in _PRIVILEGE_VALUE_OPTIONS:
            values.pop(0)
            if "=" not in argument and values:
                values.pop(0)
            continue
        if any(
            argument.startswith(option) and argument != option
            for option in ("-C", "-D", "-g", "-h", "-p", "-r", "-t", "-u")
        ):
            values.pop(0)
            continue
        if argument.startswith("-"):
            values.pop(0)
            continue
        return


def _command_basename(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", maxsplit=1)[-1].lower()
    return name[:-4] if name.endswith(".exe") else name


def _option_flags(arguments: tuple[str, ...]) -> tuple[set[str], set[str], tuple[str, ...]]:
    short: set[str] = set()
    long: set[str] = set()
    targets: list[str] = []
    options_done = False
    for argument in arguments:
        lowered = argument.lower()
        if not options_done and argument == "--":
            options_done = True
        elif not options_done and lowered.startswith("--"):
            long.add(lowered[2:].split("=", maxsplit=1)[0])
        elif not options_done and lowered.startswith("-") and len(lowered) > 1:
            short.update(lowered[1:])
        elif not options_done and lowered.startswith("/") and len(lowered) == 2:
            short.add(lowered[1])
        else:
            targets.append(argument)
    return short, long, tuple(targets)


_CONTROL_VALUE_OPTIONS: Final[frozenset[str]] = frozenset(
    {
        "--host",
        "--image",
        "--job-mode",
        "--kill-who",
        "--kill-whom",
        "--lines",
        "--machine",
        "--output",
        "--property",
        "--root",
        "--signal",
        "--state",
        "--type",
        "-H",
        "-M",
        "-n",
        "-o",
        "-p",
        "-s",
        "-t",
    }
)


def _control_command_verb(arguments: tuple[str, ...]) -> str | None:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            return arguments[index + 1].casefold() if index + 1 < len(arguments) else None
        option_name = argument.split("=", maxsplit=1)[0]
        if option_name in _CONTROL_VALUE_OPTIONS and "=" not in argument:
            index += 2
            continue
        if argument.startswith("-"):
            index += 1
            continue
        return argument.casefold()
    return None


_RM_LONG_OPTIONS: Final[frozenset[str]] = frozenset(
    {
        "dir",
        "force",
        "help",
        "interactive",
        "no-preserve-root",
        "one-file-system",
        "preserve-root",
        "recursive",
        "verbose",
        "version",
    }
)


def _is_unambiguous_rm_long_option(argument: str, canonical: str) -> bool:
    if not argument.startswith("--") or "=" in argument:
        return False
    requested = argument[2:].casefold()
    matches = tuple(option for option in _RM_LONG_OPTIONS if option.startswith(requested))
    return len(matches) == 1 and matches[0] == canonical


def _deletion_facts(
    invocation: tuple[str, ...],
) -> tuple[bool, bool, tuple[str, ...]] | None:
    name = _command_basename(invocation[0])
    if name not in {"rm", "remove-item", "ri", "rd", "rmdir", "del", "erase"}:
        return None
    recursive = False
    force = False
    targets: list[str] = []
    options_done = False
    arguments = invocation[1:]
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        lowered = argument.lower()
        if not options_done and argument == "--":
            options_done = True
        elif not options_done and lowered in {"-path", "-literalpath"}:
            if index + 1 < len(arguments):
                targets.append(arguments[index + 1])
                index += 1
        elif not options_done and (
            attached := _powershell_attached_parameter(argument, ("path", "literalpath"))
        ) is not None:
            targets.append(attached)
        elif not options_done and (
            lowered in {"-r", "-s", "/s", "--recursive", "--recurse"}
            or (name == "rm" and _is_unambiguous_rm_long_option(lowered, "recursive"))
        ):
            recursive = True
        elif not options_done and (
            lowered in {"-f", "-q", "/f", "/q", "--force"}
            or (name == "rm" and _is_unambiguous_rm_long_option(lowered, "force"))
        ):
            force = True
        elif not options_done and _powershell_switch_enabled(argument, "recurse"):
            recursive = True
        elif not options_done and _powershell_switch_enabled(argument, "force"):
            force = True
        elif not options_done and lowered.startswith("-") and not lowered.startswith("--"):
            short_flags = lowered[1:]
            if len(short_flags) <= 3 and set(short_flags) <= {"r", "s", "f", "q"}:
                recursive = recursive or bool({"r", "s"} & set(short_flags))
                force = force or bool({"f", "q"} & set(short_flags))
            else:
                targets.append(argument)
        else:
            targets.append(argument)
        index += 1
    return recursive, force, _expand_target_list(targets)


def _powershell_switch_enabled(argument: str, canonical: str) -> bool:
    if not argument.startswith("-") or argument.startswith("--"):
        return False
    body = argument[1:]
    name, separator, value = body.partition(":")
    if not name or not canonical.startswith(name.casefold()):
        return False
    return not separator or value.casefold() not in {"$false", "false", "0"}


def _powershell_attached_parameter(
    argument: str,
    canonical_names: tuple[str, ...],
) -> str | None:
    if not argument.startswith("-") or argument.startswith("--"):
        return None
    body = argument[1:]
    positions = [position for position in (body.find(":"), body.find("=")) if position >= 0]
    if not positions:
        return None
    position = min(positions)
    name = body[:position].casefold()
    matches = [candidate for candidate in canonical_names if candidate.startswith(name)]
    if len(matches) != 1:
        return None
    return body[position + 1 :]


def _is_dynamic_target(target: str) -> bool:
    return any(marker in target for marker in ("$", "`", "*", "?", "$(", "@(", "(", ")"))


def _expand_target_list(targets: list[str]) -> tuple[str, ...]:
    return tuple(
        value.strip()
        for target in targets
        for value in target.split(",")
        if value.strip()
    )


def _path_option_targets(arguments: tuple[str, ...]) -> tuple[str, ...]:
    targets: list[str] = []
    path_options = {"-path", "--path", "-literalpath", "--literalpath", "-filepath", "--filepath"}
    index = 0
    while index < len(arguments):
        lowered = arguments[index].lower()
        if lowered in path_options and index + 1 < len(arguments):
            targets.append(arguments[index + 1])
            index += 2
            continue
        attached = _powershell_attached_parameter(
            arguments[index],
            ("path", "literalpath", "filepath"),
        )
        if attached is not None:
            targets.append(attached)
            index += 1
            continue
        if not arguments[index].startswith("-"):
            targets.append(arguments[index])
        index += 1
    return _expand_target_list(targets)


def _is_broad_target(target: str) -> bool:
    raw = target.strip().casefold().strip("'\"")
    if _is_broad_git_pathspec(raw):
        return True
    lowered = _unwrap_target_expression(target)
    if _is_broad_git_pathspec(lowered):
        return True
    return _is_broad_path_expression(lowered)


def _is_broad_path_expression(value: str) -> bool:
    wildcard = re.search(r"[*?[]", value)
    if wildcard is not None:
        value = value[: wildcard.start()]
        if not value:
            return True
    return _is_broad_path_base(value)


def _is_broad_git_pathspec(value: str) -> bool:
    facts = _git_pathspec_facts(value)
    return facts is not None and (not facts[1] or _is_broad_path_expression(facts[1]))


def _git_pathspec_facts(value: str) -> tuple[frozenset[str], str] | None:
    if value.startswith(":("):
        closing = value.find(")", 2)
        if closing < 0:
            return None
        magic = frozenset(part for part in value[2:closing].split(",") if part)
        return magic, value[closing + 1 :]
    if value.startswith(":/"):
        return frozenset({"top"}), value[2:]
    if value.startswith(":!") or value.startswith(":^"):
        return frozenset({"exclude"}), value[2:]
    return None


def _is_exclusion_git_pathspec(value: str) -> bool:
    facts = _git_pathspec_facts(value.strip().casefold().strip("'\""))
    return facts is not None and bool({"exclude", "!", "^"} & facts[0])


def _unwrap_target_expression(target: str) -> str:
    value = target.strip().casefold()
    previous = None
    while value != previous:
        previous = value
        value = value.strip().strip("'\"").strip()
        if value.startswith(("$(", "@(")):
            value = value[2:].lstrip()
        elif value.startswith("("):
            value = value[1:].lstrip()
        cast = re.match(r"^\[[a-z_][a-z0-9_.]*\]\s*\(?", value)
        if cast is not None:
            value = value[cast.end() :].lstrip()
        value = re.sub(
            r"^(?:[a-z0-9_.]+\\)?filesystem::",
            "",
            value,
            count=1,
        )
        value = value.rstrip().rstrip(")").rstrip()
    return value


def _is_broad_path_base(value: str) -> bool:
    if not value:
        return False
    windows = ntpath.normpath(value.replace("/", "\\"))
    posix = posixpath.normpath(value.replace("\\", "/"))
    if windows in {".", "..", "\\", "~"} or posix in {".", "..", "/", "~"}:
        return True
    if windows.rstrip("\\") in {
        "$home",
        "${home}",
        "$env:userprofile",
        "${env:userprofile}",
        "$env:home",
        "${env:home}",
        "%userprofile%",
    }:
        return True
    drive, tail = ntpath.splitdrive(windows)
    if drive and tail in {"", "\\"}:
        return True
    if re.fullmatch(r"/(?:home|users)/[^/]+", posix) is not None:
        return True
    return re.fullmatch(r"[a-z]:\\users\\[^\\]+", windows) is not None


def _is_device_path(value: str) -> bool:
    lowered = value.lower()
    if lowered.startswith("of="):
        lowered = lowered[3:]
    return (
        re.fullmatch(r"/dev/(?:sd[a-z]\d*|nvme\w+|mmcblk\w+)", lowered) is not None
        or re.fullmatch(r"\\\\\.\\physicaldrive\d+", lowered) is not None
    )


def _overwrites_device(invocation: tuple[str, ...]) -> bool:
    name = _command_basename(invocation[0])
    arguments = invocation[1:]
    if name == "dd" and any(
        argument.lower().startswith("of=") and _is_device_path(argument)
        for argument in arguments
    ):
        return True
    attached_targets = tuple(
        target
        for argument in arguments
        if (
            target := _powershell_attached_parameter(
                argument,
                ("path", "literalpath", "filepath"),
            )
        )
        is not None
    )
    if name in {
        "set-content",
        "sc",
        "add-content",
        "ac",
        "clear-content",
        "clc",
        "out-file",
        "tee",
    } and any(
        _is_device_path(argument) for argument in (*arguments, *attached_targets)
    ):
        return True
    return any(
        argument in {">", ">>"}
        and index + 1 < len(arguments)
        and _is_device_path(arguments[index + 1])
        for index, argument in enumerate(arguments)
    )


def _git_invocation(invocation: tuple[str, ...]) -> tuple[str, tuple[str, ...]] | None:
    if _command_basename(invocation[0]) != "git":
        return None
    index = 1
    value_options = {"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--config-env"}
    while index < len(invocation):
        argument = invocation[index]
        if argument in value_options:
            index += 2
        elif argument.startswith("-C") and argument != "-C":
            index += 1
        elif argument.startswith("-"):
            index += 1
        else:
            return argument.lower(), invocation[index + 1 :]
    return None


def _git_alias_command(invocation: tuple[str, ...], subcommand: str) -> str | None:
    aliases: dict[str, str] = {}
    index = 1
    value_options = {"-C", "--git-dir", "--work-tree", "--namespace", "--config-env"}
    while index < len(invocation):
        argument = invocation[index]
        setting: str | None = None
        if argument == "-c" and index + 1 < len(invocation):
            setting = invocation[index + 1]
            index += 2
        elif argument.startswith("-c") and argument != "-c":
            setting = argument[2:].lstrip("=")
            index += 1
        elif argument in value_options:
            index += 2
        elif argument.startswith("-C") and argument != "-C":
            index += 1
        elif argument.startswith("-"):
            index += 1
        else:
            break
        if setting is None or "=" not in setting:
            continue
        key, value = setting.split("=", maxsplit=1)
        if key.casefold().startswith("alias."):
            aliases[key[6:].casefold()] = value
    body = aliases.get(subcommand.casefold())
    if body is None:
        return None
    return body[1:].lstrip() if body.startswith("!") else f"git {body}"


def _git_clean_is_catastrophic(arguments: tuple[str, ...]) -> bool:
    short, long, _targets = _option_flags(arguments)
    force = "f" in short or "force" in long
    directories = "d" in short or "directories" in long
    return force and directories and "x" in short


def _git_checkout_is_broad_force(arguments: tuple[str, ...]) -> bool:
    short, long, _targets = _option_flags(arguments)
    if "f" not in short and "force" not in long:
        return False
    if "--" not in arguments:
        return True
    pathspecs = arguments[arguments.index("--") + 1 :]
    return _git_pathspecs_are_broad(pathspecs)


def _git_restore_is_broad(arguments: tuple[str, ...]) -> bool:
    return _git_pathspecs_are_broad(_git_restore_pathspecs(arguments))


def _git_restore_pathspecs(arguments: tuple[str, ...]) -> tuple[str, ...]:
    pathspecs: list[str] = []
    value_options = {"--source", "-s", "--pathspec-from-file"}
    options_done = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        lowered = argument.casefold()
        if not options_done and argument == "--":
            options_done = True
        elif not options_done and lowered in value_options:
            index += 1
        elif not options_done and any(
            lowered.startswith(f"{option}=") for option in value_options if option.startswith("--")
        ):
            pass
        elif not options_done and lowered.startswith("-s") and lowered != "-s":
            pass
        elif not options_done and lowered.startswith("-"):
            pass
        else:
            pathspecs.append(argument)
        index += 1
    return tuple(pathspecs)


def _git_pathspecs_are_broad(pathspecs: tuple[str, ...]) -> bool:
    return any(_is_broad_target(pathspec) for pathspec in pathspecs) or (
        bool(pathspecs) and all(_is_exclusion_git_pathspec(pathspec) for pathspec in pathspecs)
    )


def _nested_shell_command(invocation: tuple[str, ...]) -> str | None:
    name = _command_basename(invocation[0])
    selectors = {
        "bash": {"-c"},
        "sh": {"-c"},
        "pwsh": {"-command", "-c"},
        "powershell": {"-command", "-c"},
        "cmd": {"/c"},
    }
    accepted = selectors.get(name)
    if accepted is None:
        return None
    for index, argument in enumerate(invocation[1:], start=1):
        if argument.lower() in accepted and index + 1 < len(invocation):
            return invocation[index + 1]
    return None


def _unquoted_source(command: str) -> str:
    result: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(command):
        character = command[index]
        if quote is not None:
            if character == quote:
                quote = None
            result.append(" ")
        elif character in {"'", '"'}:
            quote = character
            result.append(" ")
        elif character == "#":
            while index < len(command) and command[index] != "\n":
                result.append(" ")
                index += 1
            result.append("\n")
        else:
            result.append(character)
        index += 1
    return "".join(result)


def assess_command(
    command: str,
    *,
    family: ExecShellFamily,
    syntax_confidence: ExecSyntaxConfidence,
    syntax_uncertain: bool,
    command_names: tuple[str, ...] = (),
    command_identities: tuple[ExecCommandIdentity, ...] = (),
    git_delegation_safe: bool | None = None,
    diagnostics: tuple[str, ...] = (),
    inspector_status: ExecInspectorStatus = "available",
    dynamic_constructs: tuple[ExecDynamicConstruct, ...] | None = None,
    file_accesses: tuple[ExecPathAccess, ...] | None = None,
) -> ExecAssessment:
    """Build a shared assessment from parser facts and conservative text roles."""
    names = command_names or _command_names(command)
    identities = command_identities or tuple(
        ExecCommandIdentity(
            requested=name,
            kind=(
                "cmdlet"
                if family in {"powershell", "pwsh"} and "-" in name
                else "builtin"
                if name in _BUILTINS["powershell" if family in {"powershell", "pwsh"} else "bash"]
                else "unknown"
            ),
        )
        for name in names
    )
    dynamic = (
        _dynamic_constructs(command, family=family)
        if dynamic_constructs is None
        else tuple(dynamic_constructs)
    )
    return ExecAssessment(
        syntax_confidence=syntax_confidence,
        syntax_uncertain=syntax_uncertain,
        command_identities=identities,
        file_accesses=(
            _path_accesses(command, names, family=family)
            if file_accesses is None
            else tuple(file_accesses)
        ),
        network_targets=tuple(match.group(0).rstrip(".,;:!?)]}") for match in _URL_PATTERN.finditer(command)),
        dynamic_constructs=dynamic,
        catastrophic_matches=catastrophic_matches(command),
        destructive_matches=destructive_matches(command),
        git_delegation_safe=git_delegation_safe,
        inspector_status=inspector_status,
        diagnostics=diagnostics,
    )


def _command_names(command: str) -> tuple[str, ...]:
    names: list[str] = []
    for segment in re.split(r"(?:^|[;|&\n])", command):
        candidate = segment.strip()
        candidate = re.sub(r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)+", "", candidate)
        candidate = re.sub(r"^\s*&\s*", "", candidate)
        match = re.match(r"([^\s;|&]+)", candidate)
        if match is not None:
            value = match.group(1).strip("'\"")
            if value and value not in names:
                names.append(value)
    return tuple(names)


def _dynamic_constructs(command: str, *, family: ExecShellFamily) -> tuple[ExecDynamicConstruct, ...]:
    constructs: list[ExecDynamicConstruct] = []
    if _ASSIGNMENT_PATTERN.search(command) is not None:
        constructs.append(ExecDynamicConstruct("assignment", "variable assignment"))
    if _SUBSTITUTION_PATTERN.search(command) is not None:
        constructs.append(ExecDynamicConstruct("substitution", "command or process substitution"))
    if _VARIABLE_PATTERN.search(command) is not None:
        constructs.append(ExecDynamicConstruct("variable", "variable expansion"))
    if _REDIRECTION_PATTERN.search(command) is not None:
        constructs.append(ExecDynamicConstruct("redirection", "stream redirection"))
    if (
        family in {"powershell", "pwsh"}
        and _POWERSHELL_SCRIPTBLOCK_PATTERN.search(command) is not None
    ):
        constructs.append(ExecDynamicConstruct("scriptblock", "PowerShell scriptblock"))
    if re.search(r"\b(?:eval|source|invoke-expression|start-job)\b", command, re.IGNORECASE):
        constructs.append(ExecDynamicConstruct("indirect-invocation", "indirect invocation"))
    return tuple(constructs)


def _path_accesses(
    command: str,
    names: tuple[str, ...],
    *,
    family: ExecShellFamily,
) -> tuple[ExecPathAccess, ...]:
    try:
        tokens = shlex.split(command, posix=family == "bash")
    except ValueError:
        tokens = re.findall(r"[^\s]+", command)
    writes = {
        "rm",
        "rmdir",
        "del",
        "erase",
        "touch",
        "mkdir",
        "mv",
        "cp",
        "set-content",
        "add-content",
        "clear-content",
        "new-item",
        "remove-item",
        "copy-item",
        "move-item",
    }
    first = names[0].lower() if names else ""
    role: ExecPathRole = "write" if first in writes else "read"
    accesses: list[ExecPathAccess] = []
    for token in tokens:
        clean = token.lstrip("0123456789").lstrip("><")
        if not _looks_like_path(clean):
            continue
        accesses.append(ExecPathAccess(path=clean, role=role))
    if re.search(r"(?:^|\s)\d*>>?\s*", command) is not None:
        for target in re.findall(r"(?:^|\s)\d*>>?\s*([^\s]+)", command):
            accesses.append(ExecPathAccess(path=target, role="write"))
    return tuple(dict.fromkeys(accesses))


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith(("/", "./", "../", "~", "$", "\\\\"))
        or "/" in value
        or "\\" in value
        or re.fullmatch(r"[A-Za-z]:", value) is not None
        or any(character in value for character in "*?")
    )


_BUILTINS: Final[dict[ExecShellFamily, frozenset[str]]] = {
    "bash": frozenset({"cd", "echo", "printf", "pwd", "read", "test", "true", "false"}),
    "powershell": frozenset({"echo", "cd", "pwd", "where", "write-output"}),
}


@dataclass(frozen=True, slots=True)
class _BashGrammar:
    switches: frozenset[str] = frozenset()
    values: frozenset[str] = frozenset()
    optional_values: frozenset[str] = frozenset()
    minimum_operands: int = 0
    maximum_operands: int | None = None


def _bash_grammar(
    *,
    switches: tuple[str, ...] = (),
    values: tuple[str, ...] = (),
    optional_values: tuple[str, ...] = (),
    minimum_operands: int = 0,
    maximum_operands: int | None = None,
) -> _BashGrammar:
    return _BashGrammar(
        switches=frozenset(switches),
        values=frozenset(values),
        optional_values=frozenset(optional_values),
        minimum_operands=minimum_operands,
        maximum_operands=maximum_operands,
    )


BASH_APPROVED_BUILTINS: Final[frozenset[str]] = frozenset({"pwd"})
BASH_READ_CANDIDATES: Final[frozenset[str]] = frozenset(
    {
        "pwd",
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "stat",
        "file",
        "grep",
        "rg",
        "find",
        "sort",
        "uniq",
        "cut",
        "diff",
    }
)
BASH_WRITE_CANDIDATES: Final[frozenset[str]] = frozenset(
    {"mkdir", "touch", "cp", "mv", "rm"}
)

_BASH_GRAMMAR: Final[dict[str, _BashGrammar]] = {
    "pwd": _bash_grammar(
        switches=("-L", "-P", "--logical", "--physical"),
        maximum_operands=0,
    ),
    "ls": _bash_grammar(
        switches=(
            "-a",
            "-A",
            "-d",
            "-F",
            "-h",
            "-l",
            "-R",
            "-r",
            "-S",
            "-t",
            "-U",
            "-1",
            "--all",
            "--almost-all",
            "--classify",
            "--directory",
            "--human-readable",
            "--reverse",
            "--recursive",
        ),
        values=("--format", "--quoting-style", "--sort", "--time-style"),
        optional_values=("--color",),
        maximum_operands=None,
    ),
    "cat": _bash_grammar(
        switches=(
            "-A",
            "-b",
            "-E",
            "-n",
            "-s",
            "-T",
            "-v",
            "--number",
            "--show-all",
            "--show-ends",
            "--show-nonprinting",
            "--show-tabs",
            "--squeeze-blank",
        ),
        minimum_operands=1,
    ),
    "head": _bash_grammar(
        switches=("-q", "-v", "--quiet", "--silent", "--verbose"),
        values=("-n", "-c", "--lines", "--bytes"),
        minimum_operands=1,
    ),
    "tail": _bash_grammar(
        switches=("-q", "-v", "--quiet", "--silent", "--verbose"),
        values=("-n", "-c", "--lines", "--bytes"),
        minimum_operands=1,
    ),
    "wc": _bash_grammar(
        switches=("-c", "-m", "-l", "-L", "-w", "--bytes", "--chars", "--lines", "--max-line-length", "--words"),
        minimum_operands=1,
    ),
    "stat": _bash_grammar(
        switches=("-f", "-L", "-t", "--dereference", "--file-system", "--terse"),
        values=("-c", "--format", "--printf"),
        minimum_operands=1,
    ),
    "file": _bash_grammar(
        switches=(
            "-b",
            "-c",
            "-h",
            "-i",
            "-k",
            "-L",
            "-n",
            "-N",
            "-z",
            "--brief",
            "--dereference",
            "--mime",
            "--mime-type",
            "--no-buffer",
            "--preserve-date",
            "--raw",
            "--zero",
        ),
        values=("-e", "--exclude"),
        minimum_operands=1,
    ),
    "grep": _bash_grammar(),
    "rg": _bash_grammar(),
    "find": _bash_grammar(),
    "sort": _bash_grammar(
        switches=(
            "-b",
            "-d",
            "-f",
            "-g",
            "-h",
            "-i",
            "-M",
            "-n",
            "-r",
            "-s",
            "-u",
            "-z",
            "--debug",
            "--dictionary-order",
            "--general-numeric-sort",
            "--human-numeric-sort",
            "--ignore-case",
            "--ignore-nonprinting",
            "--month-sort",
            "--numeric-sort",
            "--reverse",
            "--stable",
            "--unique",
            "--zero-terminated",
        ),
        values=("-k", "-S", "-t", "--batch-size", "--field-separator", "--key"),
        minimum_operands=1,
    ),
    "uniq": _bash_grammar(
        switches=("-c", "-d", "-D", "-i", "-u", "--all-repeated", "--count", "--ignore-case", "--repeated", "--unique"),
        values=("-f", "-s", "-w", "--fields", "--skip-chars", "--check-chars"),
        minimum_operands=1,
        maximum_operands=2,
    ),
    "cut": _bash_grammar(
        switches=("-s", "--complement", "--only-delimited"),
        values=(
            "-b",
            "-c",
            "-d",
            "-f",
            "--bytes",
            "--characters",
            "--delimiter",
            "--fields",
        ),
        minimum_operands=1,
        maximum_operands=1,
    ),
    "diff": _bash_grammar(
        switches=(
            "-a",
            "-b",
            "-B",
            "-c",
            "-d",
            "-i",
            "-N",
            "-q",
            "-r",
            "-s",
            "-t",
            "-T",
            "-u",
            "-w",
            "-W",
            "--brief",
            "--color",
            "--expand-tabs",
            "--ignore-all-space",
            "--ignore-blank-lines",
            "--ignore-case",
            "--ignore-file-name-case",
            "--ignore-space-change",
            "--minimal",
            "--new-file",
            "--recursive",
            "--report-identical-files",
            "--strip-trailing-cr",
            "--text",
            "--unified",
            "--width",
        ),
        values=("-D", "--ifdef", "--label", "--palette"),
        optional_values=("--color",),
        minimum_operands=2,
        maximum_operands=2,
    ),
    "mkdir": _bash_grammar(
        switches=("-p", "-v", "--parents", "--verbose"),
        values=("-m", "--mode"),
        minimum_operands=1,
    ),
    "touch": _bash_grammar(
        switches=("-a", "-c", "-m", "--no-create"),
        values=("-d", "-t", "--date", "--reference", "-r"),
        minimum_operands=1,
    ),
    "rm": _bash_grammar(
        switches=(
            "-d",
            "-f",
            "-i",
            "-I",
            "-r",
            "-R",
            "-v",
            "--dir",
            "--force",
            "--interactive",
            "--one-file-system",
            "--preserve-root",
            "--no-preserve-root",
            "--recursive",
            "--verbose",
        ),
        optional_values=("--interactive",),
        minimum_operands=1,
    ),
    "cp": _bash_grammar(
        switches=(
            "-a",
            "-f",
            "-i",
            "-l",
            "-n",
            "-p",
            "-P",
            "-r",
            "-R",
            "-s",
            "-u",
            "-v",
            "-T",
            "--archive",
            "--attributes-only",
            "--backup",
            "--force",
            "--interactive",
            "--link",
            "--no-clobber",
            "--no-target-directory",
            "--parents",
            "--preserve",
            "--recursive",
            "--reflink",
            "--symbolic-link",
            "--update",
            "--verbose",
        ),
        values=("-S", "-t", "--suffix", "--target-directory"),
        optional_values=("--backup", "--preserve", "--reflink"),
        minimum_operands=2,
    ),
    "mv": _bash_grammar(
        switches=(
            "-f",
            "-i",
            "-n",
            "-T",
            "-u",
            "-v",
            "--backup",
            "--force",
            "--interactive",
            "--no-clobber",
            "--no-target-directory",
            "--strip-trailing-slashes",
            "--update",
            "--verbose",
        ),
        values=("-S", "-t", "--suffix", "--target-directory"),
        optional_values=("--backup",),
        minimum_operands=2,
    ),
}

_BASH_GREP_SWITCHES: Final[frozenset[str]] = frozenset(
    {
        "-i",
        "-n",
        "-v",
        "-w",
        "-x",
        "-c",
        "-l",
        "-L",
        "-q",
        "-s",
        "-h",
        "-H",
        "-r",
        "-E",
        "-F",
        "-G",
        "-P",
        "-o",
        "--count",
        "--ignore-case",
        "--line-buffered",
        "--line-number",
        "--no-filename",
        "--no-messages",
        "--only-matching",
        "--quiet",
        "--text",
        "--with-filename",
        "--word-regexp",
    }
)
_BASH_GREP_VALUES: Final[frozenset[str]] = frozenset(
    {
        "-A",
        "-B",
        "-C",
        "-e",
        "-m",
        "--after-context",
        "--before-context",
        "--binary-files",
        "--context",
        "--directories",
        "--exclude",
        "--exclude-dir",
        "--include",
        "--max-count",
        "--regexp",
    }
)
_BASH_GREP_OPTIONAL_VALUES: Final[frozenset[str]] = frozenset(
    {"--color", "--colour"}
)
_BASH_RG_SWITCHES: Final[frozenset[str]] = frozenset(
    {
        "-i",
        "-n",
        "-v",
        "-w",
        "-x",
        "-c",
        "-l",
        "-q",
        "-s",
        "-H",
        "-F",
        "-P",
        "-o",
        "--count",
        "--files",
        "--files-with-matches",
        "--files-without-match",
        "--glob-case-insensitive",
        "--heading",
        "--hidden",
        "--ignore-case",
        "--line-buffered",
        "--line-number",
        "--no-filename",
        "--no-ignore",
        "--no-messages",
        "--only-matching",
        "--quiet",
        "--text",
        "--with-filename",
        "--word-regexp",
    }
)
_BASH_RG_VALUES: Final[frozenset[str]] = frozenset(
    {
        "-A",
        "-B",
        "-C",
        "-E",
        "-e",
        "-g",
        "-t",
        "-T",
        "--after-context",
        "--before-context",
        "--color",
        "--context",
        "--encoding",
        "--glob",
        "--max-count",
        "--regexp",
        "--type",
        "--type-not",
    }
)
_BASH_FORBIDDEN_PATTERN_OPTIONS: Final[frozenset[str]] = frozenset(
    {"-f", "--file", "--pre", "--pre-glob", "--config", "--ignore-file"}
)
_BASH_FIND_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "-delete",
        "-exec",
        "-execdir",
        "-ok",
        "-okdir",
        "-fprint",
        "-fprint0",
        "-fprintf",
        "-fls",
        "-ls",
    }
)


@dataclass(frozen=True, slots=True)
class _PowerShellGrammar:
    switches: frozenset[str] = frozenset()
    values: frozenset[str] = frozenset()
    path_roles: tuple[tuple[str, ExecPathRole], ...] = ()
    positional_roles: tuple[ExecPathRole | None, ...] = ()
    pipeline_consumes_path: bool = False

    @property
    def path_role_map(self) -> dict[str, ExecPathRole]:
        return dict(self.path_roles)


def _ps_grammar(
    *,
    switches: tuple[str, ...] = (),
    values: tuple[str, ...] = (),
    path_roles: tuple[tuple[str, ExecPathRole], ...] = (),
    positional_roles: tuple[ExecPathRole | None, ...] = (),
    pipeline_consumes_path: bool = False,
) -> _PowerShellGrammar:
    return _PowerShellGrammar(
        switches=frozenset(switches),
        values=frozenset(values) | {name for name, _role in path_roles},
        path_roles=path_roles,
        positional_roles=positional_roles,
        pipeline_consumes_path=pipeline_consumes_path,
    )


POWERSHELL_READ_CANDIDATES: Final[frozenset[str]] = frozenset(
    {
        "Get-ChildItem",
        "Get-Content",
        "Get-Item",
        "Get-Location",
        "Get-FileHash",
        "Measure-Object",
        "Select-Object",
        "Sort-Object",
        "Select-String",
        "Test-Path",
        "Resolve-Path",
        "Format-List",
        "Format-Table",
        "Out-String",
    }
)
POWERSHELL_WRITE_CANDIDATES: Final[frozenset[str]] = frozenset(
    {
        "New-Item",
        "Set-Content",
        "Add-Content",
        "Clear-Content",
        "Copy-Item",
        "Move-Item",
        "Rename-Item",
        "Remove-Item",
        "Out-File",
    }
)
GIT_READ_FORMS: Final[frozenset[str]] = frozenset(
    {"status", "diff", "log", "show", "branch", "rev-parse", "ls-files"}
)

_POWERSHELL_EXPECTED_MODULES: Final[dict[str, frozenset[str]]] = {
    name: frozenset({"Microsoft.PowerShell.Management"})
    for name in (
        "Get-ChildItem",
        "Get-Content",
        "Get-Item",
        "Get-Location",
        "Test-Path",
        "Resolve-Path",
        "New-Item",
        "Set-Content",
        "Add-Content",
        "Clear-Content",
        "Copy-Item",
        "Move-Item",
        "Rename-Item",
        "Remove-Item",
    )
}
_POWERSHELL_EXPECTED_MODULES.update(
    {
        name: frozenset({"Microsoft.PowerShell.Utility"})
        for name in (
            "Get-FileHash",
            "Measure-Object",
            "Select-Object",
            "Sort-Object",
            "Select-String",
            "Format-List",
            "Format-Table",
            "Out-String",
            "Out-File",
        )
    }
)

_POWERSHELL_GRAMMAR: Final[dict[str, _PowerShellGrammar]] = {
    "get-childitem": _ps_grammar(
        switches=("recurse", "force", "file", "directory", "name", "hidden", "system"),
        values=("filter", "include", "exclude", "depth"),
        path_roles=(("path", "read"), ("literalpath", "read")),
        positional_roles=("read",),
    ),
    "get-content": _ps_grammar(
        switches=("raw", "force", "wait"),
        values=("encoding", "delimiter", "readcount", "tail", "totalcount"),
        path_roles=(("path", "read"), ("literalpath", "read")),
        positional_roles=("read",),
        pipeline_consumes_path=True,
    ),
    "get-item": _ps_grammar(
        switches=("force",),
        path_roles=(("path", "read"), ("literalpath", "read")),
        positional_roles=("read",),
        pipeline_consumes_path=True,
    ),
    "get-location": _ps_grammar(
        switches=("stack",),
        values=("psdrive",),
    ),
    "get-filehash": _ps_grammar(
        values=("algorithm",),
        path_roles=(("path", "read"), ("literalpath", "read")),
        positional_roles=("read",),
        pipeline_consumes_path=True,
    ),
    "measure-object": _ps_grammar(
        switches=("sum", "average", "minimum", "maximum", "allstats", "line", "word", "character"),
        values=("property",),
    ),
    "select-object": _ps_grammar(
        switches=("unique", "wait"),
        values=("property", "excludeproperty", "expandproperty", "first", "last", "skip"),
        positional_roles=(None,),
    ),
    "sort-object": _ps_grammar(
        switches=("descending", "ascending", "unique", "caseSensitive".lower()),
        values=("property", "culture"),
        positional_roles=(None,),
    ),
    "select-string": _ps_grammar(
        switches=("allmatches", "casesensitive", "list", "quiet", "notmatch", "simplematch"),
        values=("pattern", "context", "encoding"),
        path_roles=(("path", "read"), ("literalpath", "read")),
        positional_roles=(None,),
        pipeline_consumes_path=False,
    ),
    "test-path": _ps_grammar(
        switches=("isvalid",),
        values=("pathtype",),
        path_roles=(("path", "read"), ("literalpath", "read")),
        positional_roles=("read",),
    ),
    "resolve-path": _ps_grammar(
        switches=("relative",),
        path_roles=(("path", "read"), ("literalpath", "read")),
        positional_roles=("read",),
    ),
    "format-list": _ps_grammar(
        switches=("force",),
        values=("property", "groupby", "view"),
        positional_roles=(None,),
    ),
    "format-table": _ps_grammar(
        switches=("autosize", "wrap", "force"),
        values=("property", "groupby", "view"),
        positional_roles=(None,),
    ),
    "out-string": _ps_grammar(
        switches=("stream",),
        values=("width",),
        positional_roles=(None,),
    ),
    "new-item": _ps_grammar(
        switches=("force",),
        values=("itemtype", "value"),
        path_roles=(
            ("path", "write"),
            ("literalpath", "write"),
            ("name", "write"),
        ),
        positional_roles=("write",),
    ),
    "set-content": _ps_grammar(
        switches=("force", "nonewline", "passThru".lower()),
        values=("value", "encoding", "delimiter"),
        path_roles=(("path", "write"), ("literalpath", "write")),
        positional_roles=("write", None),
        pipeline_consumes_path=False,
    ),
    "add-content": _ps_grammar(
        switches=("force", "nonewline", "passThru".lower()),
        values=("value", "encoding", "delimiter"),
        path_roles=(("path", "write"), ("literalpath", "write")),
        positional_roles=("write", None),
        pipeline_consumes_path=False,
    ),
    "clear-content": _ps_grammar(
        switches=("force",),
        path_roles=(("path", "write"), ("literalpath", "write")),
        positional_roles=("write",),
        pipeline_consumes_path=True,
    ),
    "copy-item": _ps_grammar(
        switches=("force", "recurse", "container", "passThru".lower()),
        path_roles=(
            ("path", "read"),
            ("literalpath", "read"),
            ("destination", "write"),
        ),
        positional_roles=("read", "write"),
        pipeline_consumes_path=True,
    ),
    "move-item": _ps_grammar(
        switches=("force", "passThru".lower()),
        path_roles=(
            ("path", "read"),
            ("literalpath", "read"),
            ("destination", "write"),
        ),
        positional_roles=("read", "write"),
        pipeline_consumes_path=True,
    ),
    "rename-item": _ps_grammar(
        switches=("force", "passThru".lower()),
        path_roles=(
            ("path", "write"),
            ("literalpath", "write"),
            ("newname", "write"),
        ),
        positional_roles=("write", None),
        pipeline_consumes_path=True,
    ),
    "remove-item": _ps_grammar(
        switches=("force", "recurse", "stream", "verbose"),
        path_roles=(("path", "write"), ("literalpath", "write")),
        positional_roles=("write",),
        pipeline_consumes_path=True,
    ),
    "out-file": _ps_grammar(
        switches=("append", "noclobber", "force", "nonewline"),
        values=("width", "encoding"),
        path_roles=(("filepath", "write"), ("literalpath", "write")),
        positional_roles=("write",),
    ),
}

_GIT_SWITCHES: Final[dict[str, frozenset[str]]] = {
    "status": frozenset(
        {"--short", "--porcelain", "--branch", "--untracked-files", "--ignored", "--ahead-behind", "-s", "-b", "-u", "-uno", "-unormal", "-uall"}
    ),
    "diff": frozenset(
        {"--cached", "--staged", "--stat", "--name-only", "--name-status", "--check", "--no-color", "--no-ext-diff", "--no-textconv", "-q"}
    ),
    "log": frozenset(
        {"--oneline", "--decorate", "--stat", "--graph", "--all", "--no-color", "-n", "--max-count", "--since", "--until"}
    ),
    "show": frozenset(
        {"--stat", "--oneline", "--no-patch", "--name-only", "--name-status", "--format", "--no-color", "-s"}
    ),
    "branch": frozenset({"--list", "--all", "--remotes", "--no-color", "--format", "-a", "-r"}),
    "rev-parse": frozenset(
        {"--show-toplevel", "--show-prefix", "--git-dir", "--is-inside-work-tree", "--is-inside-git-dir", "--abbrev-ref", "--verify", "--short"}
    ),
    "ls-files": frozenset(
        {"--cached", "--deleted", "--modified", "--others", "--stage", "--unmerged", "--killed", "--directory", "--empty-directory", "--exclude-standard", "--full-name", "--eol", "--no-empty-directory", "-c", "-d", "-m", "-o", "-s"}
    ),
}

_GIT_VALUE_OPTIONS: Final[dict[str, frozenset[str]]] = {
    "status": frozenset({"--untracked-files"}),
    "diff": frozenset(),
    "log": frozenset({"-n", "--max-count", "--since", "--until"}),
    "show": frozenset({"--format"}),
    "branch": frozenset({"--format"}),
    "rev-parse": frozenset({"--abbrev-ref", "--short"}),
    "ls-files": frozenset(),
}
_GIT_OPTIONAL_VALUE_OPTIONS: Final[frozenset[str]] = frozenset(
    {"--untracked-files", "--abbrev-ref", "--short"}
)


def classify_powershell_command(
    command: str,
    assessment: ExecAssessment,
) -> ExecGrammarClassification:
    """Classify a parseable PowerShell command against fixed code-owned grammar."""
    segments = _split_powershell_pipeline(command)
    if segments is None:
        return ExecGrammarClassification(False, "PowerShell command syntax is outside the direct grammar.")
    if len(segments) != len(assessment.command_identities):
        return ExecGrammarClassification(False, "PowerShell command identity facts are incomplete.")

    accesses: list[ExecPathAccess] = []
    for index, (tokens, identity) in enumerate(
        zip(segments, assessment.command_identities, strict=True)
    ):
        if not tokens:
            return ExecGrammarClassification(False, "PowerShell pipeline contains an empty command.")
        name = tokens[0]
        if name.casefold() in {"git", "git.exe"}:
            if index != 0:
                return ExecGrammarClassification(
                    False,
                    "Git is not the first command in the PowerShell pipeline.",
                )
            if not _is_trusted_git_identity(identity, requested=name):
                return ExecGrammarClassification(False, "The Git executable identity is not trusted.")
            if assessment.git_delegation_safe is not True:
                return ExecGrammarClassification(
                    False,
                    "Git repository configuration may delegate execution.",
                )
            git_result = _classify_git_arguments(tokens[1:])
            if not git_result.accepted:
                return git_result
            accesses.extend(git_result.file_accesses)
            continue

        spec = _POWERSHELL_GRAMMAR.get(name.casefold())
        if spec is None:
            return ExecGrammarClassification(False, "The PowerShell command is not on the fixed candidate list.")
        expected_name = next(
            candidate for candidate in POWERSHELL_READ_CANDIDATES | POWERSHELL_WRITE_CANDIDATES
            if candidate.casefold() == name.casefold()
        )
        expected_modules = _POWERSHELL_EXPECTED_MODULES[expected_name]
        if (
            identity.kind != "cmdlet"
            or identity.requested.casefold() != expected_name.casefold()
            or identity.canonical is None
            or identity.canonical.casefold() != expected_name.casefold()
            or identity.module is None
            or identity.module.casefold()
            not in {module.casefold() for module in expected_modules}
            or identity.resolved is None
            or identity.resolution_count != 1
        ):
            return ExecGrammarClassification(False, "The PowerShell command identity is not trusted.")
        if index > 0 and spec.pipeline_consumes_path:
            return ExecGrammarClassification(False, "The PowerShell path is supplied by the pipeline.")
        parsed = _parse_powershell_arguments(tokens[1:], spec)
        if not parsed.accepted:
            return parsed
        accesses.extend(parsed.file_accesses)

    return ExecGrammarClassification(True, "", tuple(dict.fromkeys(accesses)))


def classify_bash_command(
    command: str,
    assessment: ExecAssessment,
) -> ExecGrammarClassification:
    """Classify a parseable Bash command against the fixed low-permission grammar."""
    if assessment.uncertain or assessment.syntax_confidence != "high":
        return ExecGrammarClassification(False, "Bash command syntax is uncertain.")
    if assessment.dynamic_constructs:
        return ExecGrammarClassification(False, "Bash command contains dynamic syntax.")
    segments = _split_bash_pipeline(command)
    if segments is None:
        return ExecGrammarClassification(False, "Bash command syntax is outside the direct grammar.")
    if len(segments) != len(assessment.command_identities):
        return ExecGrammarClassification(False, "Bash command identity facts are incomplete.")

    accesses: list[ExecPathAccess] = []
    for index, (tokens, identity) in enumerate(
        zip(segments, assessment.command_identities, strict=True)
    ):
        if not tokens:
            return ExecGrammarClassification(False, "Bash pipeline contains an empty command.")
        name = tokens[0]
        if name in {"git", "git.exe"}:
            if index != 0:
                return ExecGrammarClassification(False, "Git is not the first command in the Bash pipeline.")
            if not _is_trusted_bash_git_identity(identity, requested=name):
                return ExecGrammarClassification(False, "The Git executable identity is not trusted.")
            if assessment.git_delegation_safe is not True:
                return ExecGrammarClassification(False, "Git repository configuration may delegate execution.")
            git_result = _classify_git_arguments(
                tuple(tokens[1:]),
                path_validator=_is_static_bash_path,
                value_validator=_is_static_bash_value,
            )
            if not git_result.accepted:
                return git_result
            accesses.extend(git_result.file_accesses)
            continue

        grammar = _BASH_GRAMMAR.get(name)
        if grammar is None:
            return ExecGrammarClassification(False, "The Bash command is not on the fixed candidate list.")
        if not _is_trusted_bash_identity(identity, requested=name):
            return ExecGrammarClassification(False, "The Bash command identity is not trusted.")
        parsed = _parse_bash_candidate(
            name,
            tuple(tokens[1:]),
            grammar,
            allow_stdin=index > 0,
        )
        if not parsed.accepted:
            return parsed
        accesses.extend(parsed.file_accesses)

    return ExecGrammarClassification(True, "", tuple(dict.fromkeys(accesses)))


def _parse_bash_candidate(
    name: str,
    arguments: tuple[str, ...],
    grammar: _BashGrammar,
    *,
    allow_stdin: bool = False,
) -> ExecGrammarClassification:
    if name in {"grep", "rg"}:
        return _parse_bash_pattern_command(name, arguments, allow_stdin=allow_stdin)
    if name == "find":
        return _parse_bash_find(arguments)
    if name in {"cp", "mv"}:
        return _parse_bash_copy_move(arguments, grammar)
    if name == "touch":
        return _parse_bash_touch(arguments, grammar)
    if name == "uniq":
        return _parse_bash_uniq(arguments, grammar, allow_stdin=allow_stdin)
    if name in BASH_WRITE_CANDIDATES:
        return _parse_bash_write(arguments, grammar)
    parsed = _parse_bash_options(arguments, grammar)
    if not parsed[0]:
        return ExecGrammarClassification(False, parsed[1])
    _options, operands = parsed[2], parsed[3]
    if len(operands) < grammar.minimum_operands and not (
        allow_stdin and not operands
    ):
        return ExecGrammarClassification(False, "The Bash command is missing a path operand.")
    if grammar.maximum_operands is not None and len(operands) > grammar.maximum_operands:
        return ExecGrammarClassification(False, "The Bash command contains an unknown operand.")
    accesses: list[ExecPathAccess] = []
    for operand in operands:
        if not _is_static_bash_path(operand):
            return ExecGrammarClassification(False, "The Bash path operand is dynamic or comes from stdin.")
        accesses.append(ExecPathAccess(path=operand, role="read"))
    return ExecGrammarClassification(True, "", tuple(accesses))


def _parse_bash_write(
    arguments: tuple[str, ...],
    grammar: _BashGrammar,
) -> ExecGrammarClassification:
    parsed = _parse_bash_options(arguments, grammar)
    if not parsed[0]:
        return ExecGrammarClassification(False, parsed[1])
    _options, operands = parsed[2], parsed[3]
    if len(operands) < grammar.minimum_operands:
        return ExecGrammarClassification(False, "The Bash write command is missing a target path.")
    if grammar.maximum_operands is not None and len(operands) > grammar.maximum_operands:
        return ExecGrammarClassification(False, "The Bash write command contains an unknown operand.")
    accesses: list[ExecPathAccess] = []
    for operand in operands:
        if not _is_static_bash_path(operand):
            return ExecGrammarClassification(False, "The Bash write path operand is dynamic.")
        accesses.append(ExecPathAccess(path=operand, role="write"))
    return ExecGrammarClassification(True, "", tuple(accesses))


def _parse_bash_copy_move(
    arguments: tuple[str, ...],
    grammar: _BashGrammar,
) -> ExecGrammarClassification:
    parsed = _parse_bash_options(arguments, grammar)
    if not parsed[0]:
        return ExecGrammarClassification(False, parsed[1])
    _accepted, _reason, options, raw_operands = parsed
    operands = list(raw_operands)
    target_option = next(
        (value for name, value in options if name in {"-t", "--target-directory"}),
        None,
    )
    if target_option is not None:
        if not operands:
            return ExecGrammarClassification(False, "The Bash copy command is missing a source path.")
        if not _is_static_bash_path(target_option):
            return ExecGrammarClassification(False, "The Bash destination path is dynamic.")
        source_paths = operands
        destination = target_option
    else:
        if len(operands) < 2:
            return ExecGrammarClassification(False, "The Bash copy command needs a source and destination.")
        source_paths = operands[:-1]
        destination = operands[-1]
    if any(not _is_static_bash_path(path) for path in (*source_paths, destination)):
        return ExecGrammarClassification(False, "The Bash copy path operand is dynamic.")
    return ExecGrammarClassification(
        True,
        "",
        tuple(
            [*(ExecPathAccess(path=path, role="read") for path in source_paths),
             ExecPathAccess(path=destination, role="write")]
        ),
    )


def _parse_bash_touch(
    arguments: tuple[str, ...],
    grammar: _BashGrammar,
) -> ExecGrammarClassification:
    parsed = _parse_bash_options(arguments, grammar)
    if not parsed[0]:
        return ExecGrammarClassification(False, parsed[1])
    _accepted, _reason, options, operands = parsed
    if len(operands) < grammar.minimum_operands:
        return ExecGrammarClassification(False, "Touch is missing a target path.")
    accesses: list[ExecPathAccess] = []
    for option, value in options:
        if option not in {"-r", "--reference"} or value is None:
            continue
        if not _is_static_bash_path(value):
            return ExecGrammarClassification(False, "The touch reference path is dynamic.")
        accesses.append(ExecPathAccess(path=value, role="read"))
    for operand in operands:
        if not _is_static_bash_path(operand):
            return ExecGrammarClassification(False, "The touch target path is dynamic.")
        accesses.append(ExecPathAccess(path=operand, role="write"))
    return ExecGrammarClassification(True, "", tuple(accesses))


def _parse_bash_uniq(
    arguments: tuple[str, ...],
    grammar: _BashGrammar,
    *,
    allow_stdin: bool,
) -> ExecGrammarClassification:
    parsed = _parse_bash_options(arguments, grammar)
    if not parsed[0]:
        return ExecGrammarClassification(False, parsed[1])
    operands = parsed[3]
    if not operands and not allow_stdin:
        return ExecGrammarClassification(False, "Uniq would consume its input from stdin.")
    if len(operands) > 2:
        return ExecGrammarClassification(False, "Uniq contains an unknown operand.")
    if any(not _is_static_bash_path(operand) for operand in operands):
        return ExecGrammarClassification(False, "A uniq path operand is dynamic.")
    accesses: list[ExecPathAccess] = []
    if operands:
        accesses.append(ExecPathAccess(path=operands[0], role="read"))
    if len(operands) == 2:
        accesses.append(ExecPathAccess(path=operands[1], role="write"))
    return ExecGrammarClassification(True, "", tuple(accesses))


def _parse_bash_pattern_command(
    name: str,
    arguments: tuple[str, ...],
    *,
    allow_stdin: bool = False,
) -> ExecGrammarClassification:
    if name == "grep":
        grammar = _bash_grammar(
            switches=tuple(_BASH_GREP_SWITCHES),
            values=tuple(_BASH_GREP_VALUES),
            optional_values=tuple(_BASH_GREP_OPTIONAL_VALUES),
        )
    else:
        grammar = _bash_grammar(
            switches=tuple(_BASH_RG_SWITCHES),
            values=tuple(_BASH_RG_VALUES),
        )
    for argument in arguments:
        option_name = argument.split("=", maxsplit=1)[0]
        if option_name in _BASH_FORBIDDEN_PATTERN_OPTIONS:
            return ExecGrammarClassification(False, "Pattern-file or external preprocessor options require confirmation.")
        if option_name.startswith("--") and any(
            marker in option_name.casefold() for marker in ("config", "preprocess", "ignore-file")
        ):
            return ExecGrammarClassification(False, "Pattern configuration options require confirmation.")
    parsed = _parse_bash_options(arguments, grammar)
    if not parsed[0]:
        return ExecGrammarClassification(False, parsed[1])
    _accepted, _reason, options, raw_operands = parsed
    operands = list(raw_operands)
    patterns = [
        value
        for option, value in options
        if option in {"-e", "--regexp"} and value is not None
    ]
    files_mode = name == "rg" and any(option == "--files" for option, _value in options)
    if files_mode and patterns:
        return ExecGrammarClassification(False, "Rg files mode cannot include a search pattern.")
    if not files_mode and not patterns and operands:
        patterns.append(operands.pop(0))
    if not patterns and not files_mode:
        return ExecGrammarClassification(False, "The pattern command is missing a fixed pattern.")
    if any(not _is_static_bash_value(pattern) for pattern in patterns):
        return ExecGrammarClassification(False, "The pattern operand is dynamic.")
    if not operands and not files_mode and not allow_stdin:
        return ExecGrammarClassification(False, "The pattern command would consume its path from stdin.")
    accesses: list[ExecPathAccess] = []
    for operand in operands:
        if not _is_static_bash_path(operand):
            return ExecGrammarClassification(False, "The pattern path operand is dynamic or comes from stdin.")
        accesses.append(ExecPathAccess(path=operand, role="read"))
    return ExecGrammarClassification(True, "", tuple(accesses))


def _parse_bash_find(arguments: tuple[str, ...]) -> ExecGrammarClassification:
    if not arguments:
        return ExecGrammarClassification(False, "Find requires a fixed starting path.")
    starts: list[str] = []
    index = 0
    while index < len(arguments) and not arguments[index].startswith("-"):
        starts.append(arguments[index])
        index += 1
    if not starts or any(not _is_static_bash_path(path) for path in starts):
        return ExecGrammarClassification(False, "Find has a dynamic starting path.")
    while index < len(arguments):
        token = arguments[index]
        if token in _BASH_FIND_ACTIONS:
            return ExecGrammarClassification(False, "Find action or delegation requires confirmation.")
        if token in {"-type"}:
            if index + 1 >= len(arguments) or arguments[index + 1] not in {"b", "c", "d", "f", "l", "p", "s"}:
                return ExecGrammarClassification(False, "Find has an unknown type operand.")
            index += 2
            continue
        if token in {"-name", "-path", "-wholename"}:
            if index + 1 >= len(arguments) or not _is_static_bash_value(arguments[index + 1]):
                return ExecGrammarClassification(False, "Find has a dynamic pattern operand.")
            index += 2
            continue
        if token in {"-maxdepth", "-mindepth"}:
            if index + 1 >= len(arguments) or not arguments[index + 1].isdigit():
                return ExecGrammarClassification(False, "Find has an invalid depth operand.")
            index += 2
            continue
        if token in {"-print", "-print0", "-P", "-a"}:
            index += 1
            continue
        return ExecGrammarClassification(False, "Find contains an unknown switch or expression.")
    return ExecGrammarClassification(
        True,
        "",
        tuple(ExecPathAccess(path=path, role="read") for path in starts),
    )


def _parse_bash_options(
    arguments: tuple[str, ...],
    grammar: _BashGrammar,
) -> tuple[bool, str, tuple[tuple[str, str | None], ...], tuple[str, ...]]:
    options: list[tuple[str, str | None]] = []
    operands: list[str] = []
    options_done = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if options_done or not argument.startswith("-") or argument == "-":
            operands.append(argument)
            index += 1
            continue
        if argument == "--":
            options_done = True
            index += 1
            continue
        if argument.startswith("--"):
            name, separator, attached = argument.partition("=")
            if name in grammar.values:
                if not separator:
                    if index + 1 >= len(arguments):
                        return False, "A Bash option is missing its operand.", (), ()
                    attached = arguments[index + 1]
                    index += 1
                if attached is None or not _is_static_bash_value(attached):
                    return False, "A Bash option operand is dynamic.", (), ()
                options.append((name, attached))
            elif name in grammar.optional_values:
                if separator and not _is_static_bash_value(attached):
                    return False, "A Bash option operand is dynamic.", (), ()
                options.append((name, attached if separator else None))
            elif name in grammar.switches and not separator:
                options.append((name, None))
            else:
                return False, "The Bash command contains an unknown or abbreviated switch.", (), ()
            index += 1
            continue
        if argument in grammar.values:
            if index + 1 >= len(arguments):
                return False, "A Bash option is missing its operand.", (), ()
            attached = arguments[index + 1]
            if not _is_static_bash_value(attached):
                return False, "A Bash option operand is dynamic.", (), ()
            options.append((argument, attached))
            index += 2
            continue
        if argument in grammar.switches:
            options.append((argument, None))
            index += 1
            continue
        if len(argument) > 2 and argument.startswith("-"):
            short_options = tuple(f"-{character}" for character in argument[1:])
            if all(option in grammar.switches for option in short_options):
                options.extend((option, None) for option in short_options)
                index += 1
                continue
        return False, "The Bash command contains an unknown or combined switch.", (), ()
    return True, "", tuple(options), tuple(operands)


def _split_bash_pipeline(command: str) -> list[tuple[str, ...]] | None:
    segments: list[tuple[str, ...]] = []
    buffer: list[str] = []
    quote: str | None = None
    escaped = False
    for index, character in enumerate(command):
        if escaped:
            buffer.append(character)
            escaped = False
            continue
        if character == "\\" and quote != "'":
            buffer.append(character)
            escaped = True
            continue
        if quote is not None:
            buffer.append(character)
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
            buffer.append(character)
            continue
        if character == "|":
            if index + 1 < len(command) and command[index + 1] == "|":
                return None
            text = "".join(buffer).strip()
            if not text:
                return None
            try:
                segments.append(tuple(shlex.split(text, comments=False, posix=True)))
            except ValueError:
                return None
            buffer.clear()
            continue
        if character in ";&<>\n(){}":
            return None
        buffer.append(character)
    if quote is not None or escaped:
        return None
    text = "".join(buffer).strip()
    if not text:
        return None
    try:
        segments.append(tuple(shlex.split(text, comments=False, posix=True)))
    except ValueError:
        return None
    return segments if all(segments) else None


def _is_static_bash_value(value: str) -> bool:
    # Expansion facts come from the Bash AST before shlex removes quote context.
    return bool(value)


def _is_static_bash_path(value: str) -> bool:
    return _is_static_bash_value(value) and value != "-"


def _is_trusted_bash_identity(
    identity: ExecCommandIdentity,
    *,
    requested: str,
) -> bool:
    if identity.requested != requested or identity.resolution_count != 1:
        return False
    if identity.kind == "builtin":
        return (
            requested in BASH_APPROVED_BUILTINS
            and identity.canonical == requested
            and identity.resolved is None
        )
    if identity.kind != "native":
        return False
    return (
        "/" not in requested
        and identity.canonical == requested
        and identity.resolved is not None
        and os.path.isabs(identity.resolved)
    )


def _is_trusted_bash_git_identity(
    identity: ExecCommandIdentity,
    *,
    requested: str,
) -> bool:
    return (
        identity.requested == requested
        and identity.kind == "native"
        and identity.resolution_count == 1
        and identity.canonical == requested
        and identity.resolved is not None
        and os.path.isabs(identity.resolved)
        and "/" not in requested
    )


def bash_git_audit_targets(
    command: str,
    cwd: str,
) -> tuple[tuple[int, str], ...] | None:
    """Return static Git identity indexes and effective POSIX directories."""
    segments = _split_bash_pipeline(command)
    if segments is None:
        return None
    targets: list[tuple[int, str]] = []
    for index, tokens in enumerate(segments):
        if not tokens or tokens[0] not in {"git", "git.exe"}:
            continue
        base = Path(cwd)
        cursor = 1
        while cursor < len(tokens) and tokens[cursor] == "-C":
            if cursor + 1 >= len(tokens) or not _is_static_bash_path(tokens[cursor + 1]):
                return None
            requested = Path(tokens[cursor + 1])
            base = requested if requested.is_absolute() else base / requested
            cursor += 2
        targets.append((index, str(base.absolute())))
    return tuple(targets)


def _is_trusted_git_identity(
    identity: ExecCommandIdentity,
    *,
    requested: str,
) -> bool:
    if (
        identity.kind != "native"
        or identity.requested.casefold() != requested.casefold()
        or identity.resolved is None
        or identity.resolution_count != 1
        or identity.canonical is None
        or identity.canonical.casefold() not in {"git", "git.exe"}
    ):
        return False
    return not identity.resolved.casefold().endswith((".cmd", ".bat", ".com"))


def _split_powershell_pipeline(command: str) -> list[tuple[str, ...]] | None:
    segments: list[tuple[str, ...]] = []
    tokens: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    pipe_pending = False
    index = 0

    def flush_token() -> None:
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    def flush_segment() -> None:
        flush_token()
        if tokens:
            segments.append(tuple(tokens))
            tokens.clear()

    if not command.strip():
        return None
    while index < len(command):
        character = command[index]
        if quote is not None:
            if (
                quote == "'"
                and character == "'"
                and index + 1 < len(command)
                and command[index + 1] == "'"
            ):
                buffer.append("'")
                index += 1
            elif character == quote:
                quote = None
            elif quote == '"' and character == "`" and index + 1 < len(command):
                buffer.append(command[index + 1])
                index += 1
            else:
                buffer.append(character)
        elif character in {"'", '"'}:
            quote = character
            pipe_pending = False
        elif character == "#":
            while index < len(command) and command[index] != "\n":
                index += 1
            continue
        elif character.isspace():
            if character == "\n" and command[index + 1 :].strip():
                return None
            flush_token()
        elif character == "|":
            flush_segment()
            if not segments or pipe_pending:
                return None
            pipe_pending = True
        elif character in ";&<>\n{}()":
            return None
        elif character == "`" and index + 1 < len(command):
            buffer.append(command[index + 1])
            pipe_pending = False
            index += 1
        else:
            buffer.append(character)
            pipe_pending = False
        index += 1
    if quote is not None or pipe_pending:
        return None
    flush_segment()
    return segments if segments and all(segments) else None


def powershell_git_audit_targets(
    command: str,
    cwd: str,
) -> tuple[tuple[int, str], ...] | None:
    """Return static Git identity indexes and effective directories for config audits."""
    segments = _split_powershell_pipeline(command)
    if segments is None:
        return None
    targets: list[tuple[int, str]] = []
    for index, tokens in enumerate(segments):
        if not tokens or tokens[0].casefold() not in {"git", "git.exe"}:
            continue
        base = Path(cwd)
        cursor = 1
        while cursor < len(tokens) and tokens[cursor] == "-C":
            if cursor + 1 >= len(tokens) or not _is_static_powershell_path(tokens[cursor + 1]):
                return None
            requested = Path(tokens[cursor + 1])
            base = requested if requested.is_absolute() else base / requested
            cursor += 2
        targets.append((index, str(base.absolute())))
    return tuple(targets)


def _parse_powershell_arguments(
    arguments: tuple[str, ...],
    spec: _PowerShellGrammar,
) -> ExecGrammarClassification:
    path_roles = spec.path_role_map
    accesses: list[ExecPathAccess] = []
    positionals: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        parameter = _powershell_parameter(token)
        if parameter is None:
            if token.startswith("-"):
                return ExecGrammarClassification(False, "The PowerShell command contains an unknown switch.")
            positionals.append(token)
            index += 1
            continue
        name, attached = parameter
        if name in spec.switches:
            if attached is not None:
                return ExecGrammarClassification(False, "A PowerShell switch has an unexpected operand.")
            index += 1
            continue
        if name not in spec.values:
            return ExecGrammarClassification(False, "The PowerShell command contains an unknown parameter.")
        if attached is None:
            if index + 1 >= len(arguments):
                return ExecGrammarClassification(False, "A PowerShell parameter is missing its operand.")
            attached = arguments[index + 1]
            index += 2
        else:
            index += 1
        role = path_roles.get(name)
        if role is not None:
            if not _is_static_powershell_path(attached):
                return ExecGrammarClassification(False, "The PowerShell path operand is dynamic.")
            accesses.append(ExecPathAccess(path=attached, role=role))
        elif not _is_static_powershell_value(attached):
            return ExecGrammarClassification(False, "The PowerShell parameter operand is dynamic.")

    if len(positionals) > len(spec.positional_roles):
        return ExecGrammarClassification(False, "The PowerShell command contains an unknown operand.")
    for value, role in zip(positionals, spec.positional_roles, strict=False):
        if role is not None:
            if not _is_static_powershell_path(value):
                return ExecGrammarClassification(False, "The PowerShell path operand is dynamic.")
            accesses.append(ExecPathAccess(path=value, role=role))
        elif not _is_static_powershell_value(value):
            return ExecGrammarClassification(False, "The PowerShell operand is dynamic.")
    return ExecGrammarClassification(True, "", tuple(accesses))


def _powershell_parameter(token: str) -> tuple[str, str | None] | None:
    if not token.startswith("-") or token == "-":
        return None
    body = token[1:]
    if not body or body.startswith("-"):
        return None
    position = body.find(":")
    if position < 0:
        return body.casefold(), None
    return body[:position].casefold(), body[position + 1 :]


def _is_static_powershell_value(value: str) -> bool:
    return bool(value) and not any(marker in value for marker in ("$", "$(", "@(", "@{", "`"))


def _is_static_powershell_path(value: str) -> bool:
    if not _is_static_powershell_value(value):
        return False
    if any(marker in value for marker in ("*", "?", "[", "]", "(", ")", ",")):
        return False
    if "::" in value and not re.match(r"^[A-Za-z]:[\\/].*", value):
        return False
    if re.match(r"^[A-Za-z][A-Za-z0-9_.-]*:[\\/].*", value) and not re.match(
        r"^[A-Za-z]:[\\/].*", value
    ):
        return False
    if ":" in value and not re.match(r"^[A-Za-z]:[\\/].*", value):
        return False
    return True


def _classify_git_arguments(
    arguments: tuple[str, ...],
    *,
    path_validator: Callable[[str], bool] | None = None,
    value_validator: Callable[[str], bool] | None = None,
) -> ExecGrammarClassification:
    is_path = _is_static_powershell_path if path_validator is None else path_validator
    is_value = _is_static_powershell_value if value_validator is None else value_validator
    values = list(arguments)
    accesses: list[ExecPathAccess] = []
    while values and values[0] == "-C":
        if len(values) < 2 or not is_path(values[1]):
            return ExecGrammarClassification(False, "Git has an invalid repository path operand.")
        accesses.append(ExecPathAccess(path=values[1], role="read"))
        values = values[2:]
    if not values or values[0].startswith("-"):
        return ExecGrammarClassification(False, "Git has no fixed read form.")
    form = values.pop(0).casefold()
    if form not in GIT_READ_FORMS:
        return ExecGrammarClassification(False, "The Git form is not approved for direct execution.")
    allowed = _GIT_SWITCHES[form]
    value_options = _GIT_VALUE_OPTIONS[form]
    saw_branch_list = False
    after_separator = False
    positional_count = 0
    index = 0
    while index < len(values):
        token = values[index]
        if after_separator:
            if not is_path(token) or not _is_plain_git_pathspec(token):
                return ExecGrammarClassification(False, "Git pathspec is dynamic.")
            if form in {"status", "diff", "ls-files"}:
                accesses.append(ExecPathAccess(path=token, role="read"))
            else:
                return ExecGrammarClassification(False, "Git has an unknown operand.")
            index += 1
            continue
        if token == "--":
            after_separator = True
            index += 1
            continue
        option_name = token.split("=", maxsplit=1)[0].casefold()
        if token.startswith("-"):
            if option_name not in allowed:
                return ExecGrammarClassification(False, "Git has an unknown switch.")
            if "=" in token:
                attached = token.split("=", maxsplit=1)[1]
                if option_name not in value_options or not is_value(attached):
                    return ExecGrammarClassification(
                        False,
                        "Git has an invalid attached switch operand.",
                    )
            if form == "branch" and option_name == "--list":
                saw_branch_list = True
            if option_name in value_options and option_name not in _GIT_OPTIONAL_VALUE_OPTIONS:
                if "=" not in token:
                    if index + 1 >= len(values) or not is_value(values[index + 1]):
                        return ExecGrammarClassification(False, "Git switch is missing its operand.")
                    index += 1
            index += 1
            continue
        positional_count += 1
        if form in {"show", "log", "rev-parse"} and positional_count <= 1:
            if not is_value(token) or not _is_plain_git_revision(token):
                return ExecGrammarClassification(False, "Git revision operand is dynamic.")
        else:
            return ExecGrammarClassification(False, "Git has an unknown operand.")
        index += 1
    if form == "branch" and not saw_branch_list:
        return ExecGrammarClassification(False, "Git branch requires the fixed --list form.")
    return ExecGrammarClassification(True, "", tuple(accesses))


def _is_plain_git_pathspec(value: str) -> bool:
    return not value.startswith(":") and not any(marker in value for marker in "*?[]")


def _is_plain_git_revision(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value) is not None


__all__ = [
    "BASH_APPROVED_BUILTINS",
    "BASH_READ_CANDIDATES",
    "BASH_WRITE_CANDIDATES",
    "EXEC_CATASTROPHIC_REASON",
    "EXEC_CONFIRMATION_REASON",
    "EXEC_DESTRUCTIVE_REASON",
    "GIT_READ_FORMS",
    "POWERSHELL_READ_CANDIDATES",
    "POWERSHELL_WRITE_CANDIDATES",
    "CatastrophicMatch",
    "ExecAssessment",
    "ExecCommandIdentity",
    "ExecDynamicConstruct",
    "ExecGrammarClassification",
    "ExecIdentityKind",
    "ExecInspectorStatus",
    "ExecOutcome",
    "ExecPathAccess",
    "ExecPathRole",
    "ExecPlatform",
    "ExecShellFamily",
    "ExecShellSelector",
    "ExecSyntaxConfidence",
    "ResolvedExecShell",
    "assess_command",
    "bash_git_audit_targets",
    "bash_recursive_forced_delete_targets",
    "catastrophic_matches",
    "classify_bash_command",
    "classify_powershell_command",
    "destructive_matches",
    "powershell_git_audit_targets",
]
