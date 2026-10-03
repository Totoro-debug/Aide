"""User Configuration generation and loading."""

import os
import re
import tempfile
import tomllib
from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from hashlib import sha256
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, NoReturn, cast
from urllib.parse import urlsplit
from uuid import uuid4

import tomlkit
from croniter import croniter  # type: ignore[import-untyped]

from omni.config.agent_home import AgentHome
from omni.errors import ErrorInfo
from omni.templates import load_template
from omni.utils.host_filesystem import HOST_FILESYSTEM

DEFAULT_CONFIG_TEMPLATE: Final = load_template("default-config.md")

type ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]
type MCPTransport = Literal["stdio", "streamable-http"]
type PermissionLevel = Literal["read-only", "workspace-write", "full-access"]
type ExecShell = Literal["auto", "powershell", "pwsh"]

_PROVIDER_ID_PATTERN: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MCP_NAME_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_ROUTE_NAMES: Final = frozenset({"default", "chat", "memory", "schedule"})
_MCP_TRANSPORTS: Final = frozenset({"stdio", "streamable-http"})
_MCP_DEFAULT_CONNECT_TIMEOUT: Final = 30
_MCP_DEFAULT_CALL_TIMEOUT: Final = 60
_MCP_MAX_TIMEOUT: Final = 600
_DEFAULT_MAX_TOOL_RESULT_CHARS: Final = 4_096
_DEFAULT_MAX_ITERATIONS: Final = 50
_DEFAULT_ENABLE_SKILL_ALWAYS_LOAD: Final = False
_DEFAULT_COMPACT_RATIO: Final = 0.9
_DEFAULT_PERMISSION_LEVEL: Final[PermissionLevel] = "workspace-write"
_DEFAULT_EXEC_SHELL: Final[ExecShell] = "auto"
_DEFAULT_MEMORY_BATCH_SIZE: Final = 10
_DEFAULT_MEMORY_SCHEDULE: Final = "0 * * * *"
_DEFAULT_REASONING_EFFORT: Final[ReasoningEffort] = "medium"
_API_KEY_FIELD_PATTERN: Final = re.compile(r"api[-_]?key", flags=re.IGNORECASE)
_TOML_KEY_SEGMENT_PATTERN: Final = r"""(?:[a-z0-9_-]+|"(?:[^"\\\r\n]|\\.)*"|'[^'\r\n]*')"""


def _toml_basic_key_character_pattern(character: str) -> str:
    codepoints = sorted({ord(character.lower()), ord(character.upper())})
    escaped = "|".join(rf"\\(?:u{codepoint:04x}|U{codepoint:08x})" for codepoint in codepoints)
    return rf"(?:{re.escape(character)}|{escaped})"


def _toml_basic_key_word_pattern(word: str) -> str:
    return "".join(_toml_basic_key_character_pattern(character) for character in word)


_TOML_BASIC_API_KEY_NAME_PATTERN: Final = (
    _toml_basic_key_character_pattern("a")
    + _toml_basic_key_character_pattern("p")
    + _toml_basic_key_character_pattern("i")
    + rf"(?:{_toml_basic_key_character_pattern('-')}|"
    + rf"{_toml_basic_key_character_pattern('_')})?"
    + _toml_basic_key_character_pattern("k")
    + _toml_basic_key_character_pattern("e")
    + _toml_basic_key_character_pattern("y")
)
_API_KEY_NAME_PATTERN: Final = (
    rf"""(?:api[-_]?key|"{_TOML_BASIC_API_KEY_NAME_PATTERN}"|'api[-_]?key')"""
)
_API_KEY_ASSIGNMENT_PREFIX_PATTERN: Final = (
    rf"\s*(?:{_TOML_KEY_SEGMENT_PATTERN}\s*\.\s*)*{_API_KEY_NAME_PATTERN}\s*=\s*"
)
_API_KEY_LINE_PATTERN: Final = re.compile(
    rf"^(?P<prefix>{_API_KEY_ASSIGNMENT_PREFIX_PATTERN})(?P<value>.*)$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_API_KEY_MULTILINE_PATTERN: Final = re.compile(
    rf"(?P<prefix>(?<![a-z0-9_-]){_API_KEY_NAME_PATTERN}\s*=\s*)"
    r"(?P<quote>\"{3}|'{3}).*?(?:(?P=quote)|\Z)",
    flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_API_KEY_STRING_ASSIGNMENT_PATTERN: Final = re.compile(
    rf"(?P<prefix>(?<![a-z0-9_-]){_API_KEY_NAME_PATTERN}\s*=\s*)"
    r"(?:\"(?!\"\")(?:[^\"\\\r\n]|\\.)*\"|'(?!'')[^'\r\n]*')",
    flags=re.IGNORECASE,
)
_REDACTED_API_KEY: Final = "***REDACTED***"
_API_KEY_UNSAFE_REMAINDER_PATTERN: Final = re.compile(
    rf"(?P<prefix>(?<![a-z0-9_-]){_API_KEY_NAME_PATTERN}\s*=)"
    rf"(?!\s*[\"']{re.escape(_REDACTED_API_KEY)}[\"'])"
    r"(?P<spacing>\s*).*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)
_SENSITIVE_CONFIGURATION_FIELDS: Final = frozenset({"headers", "env", "secret_env"})
_TOML_BASIC_SENSITIVE_FIELD_PATTERN: Final = "|".join(
    _toml_basic_key_word_pattern(field) for field in sorted(_SENSITIVE_CONFIGURATION_FIELDS)
)
_SENSITIVE_FIELD_REFERENCE_PATTERN: Final = re.compile(
    rf"(?<![a-z0-9_-])(?:{_TOML_BASIC_SENSITIVE_FIELD_PATTERN})(?![a-z0-9_-])",
    flags=re.IGNORECASE,
)
_TOML_DOTTED_KEY_PATTERN: Final = (
    rf"{_TOML_KEY_SEGMENT_PATTERN}(?:\s*\.\s*{_TOML_KEY_SEGMENT_PATTERN})*"
)
_TOML_ASSIGNMENT_PATTERN: Final = re.compile(
    rf"^(?P<prefix>\s*(?P<key>{_TOML_DOTTED_KEY_PATTERN})\s*=\s*)"
    r"(?P<value>[^\r\n]*)(?P<newline>\r?\n)?\Z",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    max_tool_result_chars: int
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    enable_skill_always_load: bool = _DEFAULT_ENABLE_SKILL_ALWAYS_LOAD
    compact_ratio: float = _DEFAULT_COMPACT_RATIO
    permission_level: PermissionLevel = _DEFAULT_PERMISSION_LEVEL
    exec_shell: ExecShell = _DEFAULT_EXEC_SHELL


@dataclass(frozen=True, slots=True)
class MemoryConfiguration:
    batch_size: int
    schedule: str


@dataclass(frozen=True, slots=True)
class ProviderConfiguration:
    provider_id: str
    protocol: str
    base_url: str
    api_key: str
    models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RouteConfiguration:
    provider_id: str
    model: str
    context_window: int
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort
    timeout: int


@dataclass(frozen=True, slots=True)
class ModelsConfiguration:
    providers: Mapping[str, ProviderConfiguration]
    routes: Mapping[str, RouteConfiguration]


@dataclass(frozen=True, slots=True)
class ResolvedModelRoute:
    requested_route: str
    selected_route: str
    provider: ProviderConfiguration
    route: RouteConfiguration
    used_default: bool


def normalize_mcp_tool_keywords(value: object) -> tuple[str, ...]:
    """Validate and canonicalize one MCP Tool keyword sequence."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("MCP Tool keywords must be an array of strings")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("MCP Tool keywords must be an array of strings")
        keyword = item.strip()
        if not keyword:
            continue
        if (
            not keyword.isascii()
            or re.search(r"[A-Za-z]", keyword) is None
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in keyword)
        ):
            raise ValueError("MCP Tool keywords must contain English terms")
        if keyword not in normalized:
            normalized.append(keyword)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class MCPServerConfiguration:
    """One validated, user-selected MCP Server configuration."""

    mcp_name: str
    enabled: bool
    transport: MCPTransport
    command: str | None = None
    args: tuple[str, ...] = ()
    cwd: Path | None = None
    url: str | None = None
    headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    connect_timeout: int = _MCP_DEFAULT_CONNECT_TIMEOUT
    call_timeout: int = _MCP_DEFAULT_CALL_TIMEOUT
    tool_keywords: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.tool_keywords, Mapping):
            raise TypeError("MCP Server tool_keywords must be a mapping")
        normalized: dict[str, tuple[str, ...]] = {}
        for remote_name, raw_keywords in self.tool_keywords.items():
            if not isinstance(remote_name, str) or not remote_name:
                raise ValueError("MCP Server tool_keywords names must be non-empty strings")
            normalized[remote_name] = normalize_mcp_tool_keywords(raw_keywords)
        object.__setattr__(self, "tool_keywords", MappingProxyType(normalized))

    def resolve_cwd(self, workspace: Path) -> Path:
        """Resolve a stdio cwd against the active Workspace."""
        if self.cwd is None or self.cwd.is_absolute():
            return self.cwd if self.cwd is not None else workspace
        return workspace / self.cwd


def _empty_mcp_servers() -> Mapping[str, MCPServerConfiguration]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class UserConfiguration:
    runtime: RuntimeConfiguration
    memory: MemoryConfiguration
    models: ModelsConfiguration
    mcp: Mapping[str, MCPServerConfiguration] = field(default_factory=_empty_mcp_servers)

    def resolve_route(self, requested_route: str) -> ResolvedModelRoute:
        """Resolve a Model Route, falling back to a usable default when permitted."""
        _require_supported_route(requested_route)

        candidate = _usable_route(self.models, requested_route)
        selected_route = requested_route
        if candidate is None and requested_route != "default":
            candidate = _usable_route(self.models, "default")
            selected_route = "default"
        if candidate is None:
            raise _route_unavailable_error(self.models)
        provider, route = candidate
        return ResolvedModelRoute(
            requested_route=requested_route,
            selected_route=selected_route,
            provider=provider,
            route=route,
            used_default=selected_route != requested_route,
        )


@dataclass(frozen=True, slots=True)
class ConfigurationDiagnostic:
    """A safe diagnostic for one ignored MCP Server configuration."""

    mcp_name: str
    reason: str

    @property
    def message(self) -> str:
        return f"MCP Server {self.mcp_name!a} ignored: {self.reason}"


@dataclass(frozen=True, slots=True)
class DefaultValueDiagnostic:
    """A safe diagnostic for a configuration field using its declared default."""

    field: str
    default_value: object

    @property
    def message(self) -> str:
        default_text = self.default_value
        if isinstance(default_text, bool):
            rendered_default = str(default_text).lower()
        elif isinstance(default_text, str):
            rendered_default = repr(default_text)
        else:
            rendered_default = str(default_text)
        return f"Configuration field {self.field!r} is invalid; using {rendered_default}."


type ConfigurationDiagnosticValue = ConfigurationDiagnostic | DefaultValueDiagnostic


@dataclass(frozen=True, slots=True)
class ConfigView:
    """A configuration path, redacted content, parse error, and safe diagnostics."""

    path: Path
    redacted_content: str
    error: ErrorInfo | None
    diagnostics: tuple[ConfigurationDiagnosticValue, ...] = ()
    effective_compact_ratio: float | None = None
    effective_permission_level: PermissionLevel | None = None
    effective_exec_shell: ExecShell | None = None

    def diagnostics_text(self) -> str:
        return "".join(f"{diagnostic.message}\n" for diagnostic in self.diagnostics)

    def effective_values_text(self) -> str:
        lines: list[str] = []
        if self.effective_compact_ratio is not None:
            lines.append(f"Effective runtime.compact_ratio: {self.effective_compact_ratio:g}\n")
        if self.effective_permission_level is not None:
            lines.append(
                f"Effective runtime.permission_level: {self.effective_permission_level}\n"
            )
        if self.effective_exec_shell is not None:
            lines.append(f"Effective runtime.exec_shell: {self.effective_exec_shell}\n")
        return "".join(lines)

    def header_text(self) -> str:
        """Render the shared CLI and Management configuration header."""
        error_text = ""
        if self.error is not None:
            error_text = f"{self.error.code}: {self.error.message}\n"
        return (
            f"{error_text}{self.effective_values_text()}{self.diagnostics_text()}"
            f"Path: {self.path}\n"
        )


@dataclass(frozen=True, slots=True)
class ConfigEditableSnapshot:
    """Safe editable configuration values exposed to an authenticated editor."""

    revision: str
    fields: Mapping[str, Mapping[str, object]]
    configuration: UserConfiguration
    diagnostics: tuple[ConfigurationDiagnosticValue, ...] = ()

    def require_valid_candidate(self) -> None:
        """Reject startup fallbacks when publishing a runtime generation."""
        _require_complete_candidate(self.configuration, self.diagnostics)


@dataclass(frozen=True, slots=True)
class ConfigWebSnapshot:
    """Safe Web projection for active, missing, and repairable configuration."""

    revision: str
    fields: Mapping[str, Mapping[str, object]]
    configuration: UserConfiguration
    state: Literal["active", "missing", "invalid", "malformed"]
    repair_required: bool
    backup_required: bool
    requires_secret_reentry: bool
    error: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ConfigEditResult:
    """The validated result of one compare-and-swap configuration edit."""

    revision: str
    fields: Mapping[str, Mapping[str, object]]
    configuration: UserConfiguration
    backup_id: str | None = None


class ConfigError(Exception):
    """A safe User Configuration error suitable for a CLI or Management view."""

    def __init__(self, error: ErrorInfo, *, field_errors: dict[str, str] | None = None) -> None:
        self.error = error
        self.field_errors = {} if field_errors is None else field_errors
        super().__init__(error.message)


class ConfigRevisionConflict(ConfigError):
    """Raised when a configuration edit was based on an older file revision."""

    def __init__(self, expected_revision: str, current_revision: str) -> None:
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        super().__init__(
            ErrorInfo(
                "config_invalid",
                "User Configuration changed before this edit was applied.",
            )
        )


class ConfigFieldError(ConfigError):
    """A validation failure for a known editable field without its supplied value."""

    def __init__(self, field_name: str, rule: str) -> None:
        super().__init__(
            ErrorInfo("config_invalid", "Review the highlighted settings."),
            field_errors={field_name: rule},
        )


def _invalid(field: str, rule: str) -> NoReturn:
    raise ConfigError(
        ErrorInfo("config_invalid", f"Configuration field '{field}' {rule}."),
        field_errors={
            field.removeprefix("config.secrets.").replace("mcp.servers.", "mcp.", 1): rule
        },
    )


def _table(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _invalid(field, "must be a table")
    return cast(dict[str, object], value)


def _required(table: Mapping[str, object], key: str, field: str) -> object:
    if key not in table:
        _invalid(field, "is required")
    return table[key]


def _require_supported_route(requested_route: str) -> None:
    if requested_route not in _ROUTE_NAMES:
        _invalid("models.routes", "was requested with an unsupported route name")


def _route_unavailable_error(models: ModelsConfiguration) -> ConfigError:
    message = "Default Model Route is unavailable."
    if "default" not in models.routes:
        message = (
            "Default Model Route is missing. Add [models.routes.default] to User Configuration."
        )
    return ConfigError(ErrorInfo("route_unavailable", message))


def _missing_default_route_error() -> ConfigError:
    return ConfigError(
        ErrorInfo(
            "route_unavailable",
            "Default Model Route is missing. Add [models.routes.default] to User Configuration.",
        ),
        field_errors={"models.routes.default": "must define a default Model Route"},
    )


def _string(value: object, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        _invalid(field, "must be a string")
    if nonempty and (not value or value != value.strip()):
        _invalid(field, "must be a nonempty string without surrounding whitespace")
    return value


def _integer(value: object, field: str, minimum: int, maximum: int | None = None) -> int:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= minimum
        and (maximum is None or value <= maximum)
    )
    if not valid:
        rule = (
            f"must be an integer from {minimum} to {maximum}"
            if maximum is not None
            else f"must be an integer at least {minimum}"
        )
        _invalid(field, rule)
    return cast(int, value)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _invalid(field, "must be a boolean")
    return value


def _number(value: object, field: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not minimum <= value <= maximum
    ):
        _invalid(field, f"must be a finite number from {minimum:g} to {maximum:g}")
    return float(value)


def _has_absolute_http_url(value: str) -> bool:
    if not value or any(character.isspace() for character in value):
        return False
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and parsed.hostname is not None


def _usable_route(
    models: ModelsConfiguration, route_name: str
) -> tuple[ProviderConfiguration, RouteConfiguration] | None:
    route = models.routes.get(route_name)
    if route is None:
        return None
    provider = models.providers.get(route.provider_id)
    if provider is None or provider.protocol not in {"anthropic", "openai-compatible"}:
        return None
    if (
        not _has_absolute_http_url(provider.base_url)
        or not provider.api_key.strip()
        or not provider.models
        or route.model not in provider.models
    ):
        return None
    return provider, route


def _redact_parsed_content(content: str) -> str:
    source_document = tomlkit.parse(content)
    _redact_api_key_fields(source_document)
    return tomlkit.dumps(source_document)


def _redact_api_key_fields(value: object) -> None:
    if isinstance(value, MutableSequence):
        for item in value:
            _redact_api_key_fields(item)
        return
    if not isinstance(value, MutableMapping):
        return
    for field_name, item in tuple(value.items()):
        if (
            isinstance(field_name, str)
            and _API_KEY_FIELD_PATTERN.fullmatch(field_name)
            and item != ""
        ):
            value[field_name] = _REDACTED_API_KEY
            continue
        if isinstance(field_name, str) and field_name.lower() in _SENSITIVE_CONFIGURATION_FIELDS:
            if isinstance(item, MutableMapping):
                for header_name in tuple(item):
                    item[header_name] = _REDACTED_API_KEY
            else:
                value[field_name] = _REDACTED_API_KEY
            continue
        _redact_api_key_fields(item)


def _redact_unparsed_content(content: str) -> str:
    def redact_line(match: re.Match[str]) -> str:
        return f'{match.group("prefix")}"{_REDACTED_API_KEY}"'

    def redact_remainder(match: re.Match[str]) -> str:
        return f'{match.group("prefix")}{match.group("spacing")}"{_REDACTED_API_KEY}"'

    without_multiline_keys = _API_KEY_MULTILINE_PATTERN.sub(redact_line, content)
    without_string_keys = _API_KEY_STRING_ASSIGNMENT_PATTERN.sub(
        redact_line,
        without_multiline_keys,
    )
    without_unsafe_remainder = _API_KEY_UNSAFE_REMAINDER_PATTERN.sub(
        redact_remainder,
        without_string_keys,
    )
    without_api_keys = _API_KEY_LINE_PATTERN.sub(redact_line, without_unsafe_remainder)
    return _redact_sensitive_content(without_api_keys)


def _single_line_safe_text(value: str) -> str:
    return "".join(
        character if character.isprintable() else ascii(character)[1:-1] for character in value
    )


def _contains_sensitive_configuration_field(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            (isinstance(field_name, str) and field_name.lower() in _SENSITIVE_CONFIGURATION_FIELDS)
            or _contains_sensitive_configuration_field(item)
            for field_name, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_sensitive_configuration_field(item) for item in value)
    return False


def _split_ignorable_prefix(value: str) -> tuple[str, str]:
    index = 0
    while index < len(value) and (value[index].isspace() or not value[index].isprintable()):
        index += 1
    return value[:index], value[index:]


def _sensitive_table_header(line: str) -> bool | None:
    _, candidate = _split_ignorable_prefix(line)
    if not candidate.startswith("["):
        return None
    try:
        document = tomllib.loads(candidate)
    except tomllib.TOMLDecodeError:
        return _SENSITIVE_FIELD_REFERENCE_PATTERN.search(candidate) is not None
    return _contains_sensitive_configuration_field(document)


def _sensitive_assignment(match: re.Match[str]) -> bool:
    assignment = f"{match.group('key')} = {match.group('value')}"
    try:
        document = tomllib.loads(assignment)
    except tomllib.TOMLDecodeError:
        return _SENSITIVE_FIELD_REFERENCE_PATTERN.search(assignment) is not None
    return _contains_sensitive_configuration_field(document)


def _complete_toml_value(value: str) -> bool:
    try:
        tomllib.loads(f"value = {value}")
    except tomllib.TOMLDecodeError:
        return False
    return True


def _redacted_line(line: str) -> str:
    newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
    return f'"{_REDACTED_API_KEY}"{newline}'


def _redact_sensitive_content(content: str) -> str:
    lines: list[str] = []
    sensitive_table = False
    pending_sensitive_value: list[str] | None = None
    for line in content.splitlines(keepends=True):
        if pending_sensitive_value is not None:
            pending_sensitive_value.append(line)
            lines.append(_redacted_line(line))
            if _complete_toml_value("".join(pending_sensitive_value)):
                pending_sensitive_value = None
            continue

        table_header = _sensitive_table_header(line)
        if table_header is not None:
            sensitive_table = table_header
            lines.append(line)
            continue
        ignorable_prefix, assignment_line = _split_ignorable_prefix(line)
        if not sensitive_table or not line.strip() or line.lstrip().startswith("#"):
            assignment = _TOML_ASSIGNMENT_PATTERN.fullmatch(assignment_line)
            if assignment is None:
                if (
                    "=" in assignment_line
                    and _SENSITIVE_FIELD_REFERENCE_PATTERN.search(assignment_line) is not None
                ):
                    lines.append(_redacted_line(line))
                    pending_sensitive_value = [assignment_line]
                    continue
                lines.append(line)
                continue
            if not _sensitive_assignment(assignment):
                lines.append(line)
                continue
            lines.append(
                f'{ignorable_prefix}{assignment.group("prefix")}"{_REDACTED_API_KEY}"'
                f"{assignment.group('newline') or ''}"
            )
            if not _complete_toml_value(assignment.group("value")):
                pending_sensitive_value = [
                    assignment.group("value"),
                    assignment.group("newline") or "",
                ]
            continue

        assignment = _TOML_ASSIGNMENT_PATTERN.fullmatch(assignment_line)
        if assignment is None:
            lines.append(_redacted_line(line))
            continue
        lines.append(
            f'{ignorable_prefix}{assignment.group("prefix")}"{_REDACTED_API_KEY}"'
            f"{assignment.group('newline') or ''}"
        )
    return "".join(lines)


_MISSING: Final = object()


def _defaulted[DefaultableValue](
    table: Mapping[str, object],
    key: str,
    *,
    field: str,
    default: DefaultableValue,
    parse: Callable[[object], DefaultableValue | None],
    diagnostics: list[ConfigurationDiagnosticValue] | None,
) -> DefaultableValue:
    value = table.get(key, _MISSING)
    if value is _MISSING:
        return default
    parsed = parse(value)
    if parsed is not None:
        return parsed
    if diagnostics is not None:
        diagnostics.append(
            DefaultValueDiagnostic(
                field=field,
                default_value=default,
            )
        )
    return default


def _parse_default_integer(value: object, minimum: int, maximum: int | None) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        return None
    return value


def _parse_default_boolean(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _parse_default_compact_ratio(value: object) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0.5 <= value <= 0.95
    ):
        return None
    return float(value)


def _parse_default_schedule(value: object) -> str | None:
    if not isinstance(value, str) or len(value.split()) != 5 or not croniter.is_valid(value):
        return None
    return value


def _parse_default_permission_level(value: object) -> PermissionLevel | None:
    if not isinstance(value, str) or value not in {"read-only", "workspace-write", "full-access"}:
        return None
    return cast(PermissionLevel, value)


def _parse_default_exec_shell(value: object) -> ExecShell | None:
    if not isinstance(value, str) or value not in {"auto", "powershell", "pwsh"}:
        return None
    return cast(ExecShell, value)


def _parse_default_reasoning_effort(value: object) -> ReasoningEffort | None:
    if not isinstance(value, str) or value not in {"low", "medium", "high", "xhigh", "max"}:
        return None
    return cast(ReasoningEffort, value)


def _parse_runtime(
    document: Mapping[str, object],
    *,
    diagnostics: list[ConfigurationDiagnosticValue] | None,
) -> RuntimeConfiguration:
    table = _table(document.get("runtime", {}), "runtime")
    return RuntimeConfiguration(
        max_tool_result_chars=_defaulted(
            table,
            "max_tool_result_chars",
            field="runtime.max_tool_result_chars",
            default=_DEFAULT_MAX_TOOL_RESULT_CHARS,
            parse=lambda value: _parse_default_integer(value, 1000, 1_000_000),
            diagnostics=diagnostics,
        ),
        max_iterations=_defaulted(
            table,
            "max_iterations",
            field="runtime.max_iterations",
            default=_DEFAULT_MAX_ITERATIONS,
            parse=lambda value: _parse_default_integer(value, 50, None),
            diagnostics=diagnostics,
        ),
        enable_skill_always_load=_defaulted(
            table,
            "enable_skill_always_load",
            field="runtime.enable_skill_always_load",
            default=_DEFAULT_ENABLE_SKILL_ALWAYS_LOAD,
            parse=_parse_default_boolean,
            diagnostics=diagnostics,
        ),
        compact_ratio=_defaulted(
            table,
            "compact_ratio",
            field="runtime.compact_ratio",
            default=_DEFAULT_COMPACT_RATIO,
            parse=_parse_default_compact_ratio,
            diagnostics=diagnostics,
        ),
        permission_level=_defaulted(
            table,
            "permission_level",
            field="runtime.permission_level",
            default=_DEFAULT_PERMISSION_LEVEL,
            parse=_parse_default_permission_level,
            diagnostics=diagnostics,
        ),
        exec_shell=_defaulted(
            table,
            "exec_shell",
            field="runtime.exec_shell",
            default=_DEFAULT_EXEC_SHELL,
            parse=_parse_default_exec_shell,
            diagnostics=diagnostics,
        ),
    )


def _parse_memory(
    document: Mapping[str, object],
    *,
    diagnostics: list[ConfigurationDiagnosticValue] | None,
) -> MemoryConfiguration:
    table = _table(document.get("memory", {}), "memory")
    return MemoryConfiguration(
        batch_size=_defaulted(
            table,
            "batch_size",
            field="memory.batch_size",
            default=_DEFAULT_MEMORY_BATCH_SIZE,
            parse=lambda value: _parse_default_integer(value, 1, 1000),
            diagnostics=diagnostics,
        ),
        schedule=_defaulted(
            table,
            "schedule",
            field="memory.schedule",
            default=_DEFAULT_MEMORY_SCHEDULE,
            parse=_parse_default_schedule,
            diagnostics=diagnostics,
        ),
    )


def _parse_provider(provider_id: str, value: object) -> ProviderConfiguration:
    prefix = f"models.providers.{provider_id}"
    if not _PROVIDER_ID_PATTERN.fullmatch(provider_id):
        _invalid(prefix, "must use a lowercase kebab-case provider ID")
    table = _table(value, prefix)
    models_value = _required(table, "models", f"{prefix}.models")
    if not isinstance(models_value, list):
        _invalid(f"{prefix}.models", "must be an array of unique nonempty model IDs")
    model_items = cast(list[object], models_value)
    models: list[str] = []
    for model_value in model_items:
        model = _string(model_value, f"{prefix}.models", nonempty=True)
        if model in models:
            _invalid(f"{prefix}.models", "must contain unique model IDs")
        models.append(model)
    return ProviderConfiguration(
        provider_id=provider_id,
        protocol=_string(_required(table, "protocol", f"{prefix}.protocol"), f"{prefix}.protocol"),
        base_url=_string(_required(table, "base_url", f"{prefix}.base_url"), f"{prefix}.base_url"),
        api_key=_string(_required(table, "api_key", f"{prefix}.api_key"), f"{prefix}.api_key"),
        models=tuple(models),
    )


def _parse_route(
    route_name: str,
    value: object,
    *,
    diagnostics: list[ConfigurationDiagnosticValue] | None,
) -> RouteConfiguration:
    prefix = f"models.routes.{route_name}"
    if route_name not in _ROUTE_NAMES:
        _invalid(prefix, "is not a supported Model Route")
    table = _table(value, prefix)
    provider_id = _string(
        _required(table, "provider_id", f"{prefix}.provider_id"),
        f"{prefix}.provider_id",
        nonempty=True,
    )
    if not _PROVIDER_ID_PATTERN.fullmatch(provider_id):
        _invalid(f"{prefix}.provider_id", "must be a lowercase kebab-case provider ID")
    context_window = _integer(
        _required(table, "context_window", f"{prefix}.context_window"),
        f"{prefix}.context_window",
        1024,
        10_000_000,
    )
    max_output = _integer(
        _required(table, "max_output", f"{prefix}.max_output"),
        f"{prefix}.max_output",
        1,
        9_999_999,
    )
    if max_output >= context_window:
        _invalid(f"{prefix}.max_output", "must be less than context_window")
    return RouteConfiguration(
        provider_id=provider_id,
        model=_string(
            _required(table, "model", f"{prefix}.model"),
            f"{prefix}.model",
            nonempty=True,
        ),
        context_window=context_window,
        max_output=max_output,
        temperature=_number(
            _required(table, "temperature", f"{prefix}.temperature"),
            f"{prefix}.temperature",
            0,
            2,
        ),
        reasoning_effort=_defaulted(
            table,
            "reasoning_effort",
            field=f"{prefix}.reasoning_effort",
            default=_DEFAULT_REASONING_EFFORT,
            parse=_parse_default_reasoning_effort,
            diagnostics=diagnostics,
        ),
        timeout=_integer(
            _required(table, "timeout", f"{prefix}.timeout"),
            f"{prefix}.timeout",
            1,
            600,
        ),
    )


def _parse_models(
    document: Mapping[str, object],
    *,
    diagnostics: list[ConfigurationDiagnosticValue] | None,
) -> ModelsConfiguration:
    models_value = document.get("models", {})
    table = _table(models_value, "models")
    provider_tables = _table(table.get("providers", {}), "models.providers")
    route_tables = _table(table.get("routes", {}), "models.routes")
    providers = {
        provider_id: _parse_provider(provider_id, provider)
        for provider_id, provider in provider_tables.items()
    }
    routes = {
        route_name: _parse_route(route_name, route, diagnostics=diagnostics)
        for route_name, route in route_tables.items()
        if route_name in _ROUTE_NAMES
    }
    return ModelsConfiguration(
        providers=MappingProxyType(providers),
        routes=MappingProxyType(routes),
    )


def _parse_string_array(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _invalid(field, "must be an array of strings")
    items = cast(list[object], value)
    return tuple(_string(item, field) for item in items)


def _parse_mcp_headers(value: object, field: str) -> Mapping[str, str]:
    table = _table(value, field)
    headers: dict[str, str] = {}
    for header_name, header_value in table.items():
        if (
            not isinstance(header_name, str)
            or not header_name
            or header_name != header_name.strip()
        ):
            _invalid(field, "must contain nonempty header names without surrounding whitespace")
        headers[header_name] = _string(header_value, f"{field}.{header_name}")
    return MappingProxyType(headers)


def _parse_mcp_server(mcp_name: str, value: object) -> MCPServerConfiguration:
    valid_name = _MCP_NAME_PATTERN.fullmatch(mcp_name) is not None
    prefix = f"mcp.servers.{mcp_name if valid_name else ascii(mcp_name)}"
    if not valid_name:
        _invalid(prefix, "must use a lowercase name with up to 64 letters, digits, '_' or '-'")
    table = _table(value, prefix)
    for field_name in ("env", "secret_env"):
        if field_name in table:
            _invalid(f"{prefix}.{field_name}", "is not recognized")
    transport = _string(
        _required(table, "transport", f"{prefix}.transport"),
        f"{prefix}.transport",
    )
    if transport not in _MCP_TRANSPORTS:
        _invalid(
            f"{prefix}.transport",
            "must be either 'stdio' or 'streamable-http'",
        )
    enabled = _boolean(table.get("enabled", False), f"{prefix}.enabled")
    connect_timeout = _integer(
        table.get("connect_timeout", _MCP_DEFAULT_CONNECT_TIMEOUT),
        f"{prefix}.connect_timeout",
        1,
        _MCP_MAX_TIMEOUT,
    )
    call_timeout = _integer(
        table.get("call_timeout", _MCP_DEFAULT_CALL_TIMEOUT),
        f"{prefix}.call_timeout",
        1,
        _MCP_MAX_TIMEOUT,
    )
    keyword_field = f"{prefix}.tool_keywords"
    keyword_table = _table(table.get("tool_keywords", {}), keyword_field)
    parsed_keywords: dict[str, tuple[str, ...]] = {}
    for remote_name, raw_keywords in keyword_table.items():
        if not isinstance(remote_name, str) or not remote_name:
            _invalid(keyword_field, "must contain nonempty remote Tool names")
        remote_field = f"{keyword_field}.{remote_name}"
        try:
            parsed_keywords[remote_name] = normalize_mcp_tool_keywords(raw_keywords)
        except TypeError:
            _invalid(remote_field, "must be an array of strings")
        except ValueError:
            _invalid(remote_field, "must contain English terms")
    tool_keywords = MappingProxyType(parsed_keywords)

    if transport == "stdio":
        for field_name in ("url", "headers"):
            if field_name in table:
                _invalid(
                    f"{prefix}.{field_name}",
                    "is only valid for the streamable-http transport",
                )
        args_value = table.get("args", [])
        cwd_value = table.get("cwd")
        return MCPServerConfiguration(
            mcp_name=mcp_name,
            enabled=enabled,
            transport="stdio",
            command=_string(
                _required(table, "command", f"{prefix}.command"),
                f"{prefix}.command",
                nonempty=True,
            ),
            args=_parse_string_array(args_value, f"{prefix}.args"),
            cwd=(
                Path(_string(cwd_value, f"{prefix}.cwd", nonempty=True))
                if cwd_value is not None
                else None
            ),
            connect_timeout=connect_timeout,
            call_timeout=call_timeout,
            tool_keywords=tool_keywords,
        )

    for field_name in ("command", "args", "cwd"):
        if field_name in table:
            _invalid(
                f"{prefix}.{field_name}",
                "is only valid for the stdio transport",
            )
    url = _string(_required(table, "url", f"{prefix}.url"), f"{prefix}.url", nonempty=True)
    if not _has_absolute_http_url(url):
        _invalid(f"{prefix}.url", "must be an absolute HTTP or HTTPS URL")
    return MCPServerConfiguration(
        mcp_name=mcp_name,
        enabled=enabled,
        transport="streamable-http",
        url=url,
        headers=_parse_mcp_headers(
            table.get("headers", {}),
            f"{prefix}.headers",
        ),
        connect_timeout=connect_timeout,
        call_timeout=call_timeout,
        tool_keywords=tool_keywords,
    )


def _parse_mcp(
    document: Mapping[str, object],
    *,
    diagnostics: list[ConfigurationDiagnosticValue] | None = None,
) -> Mapping[str, MCPServerConfiguration]:
    table = _table(document.get("mcp", {}), "mcp")
    servers = _table(table.get("servers", {}), "mcp.servers")
    parsed: dict[str, MCPServerConfiguration] = {}
    for mcp_name, value in servers.items():
        try:
            parsed[mcp_name] = _parse_mcp_server(mcp_name, value)
        except ConfigError as error:
            if diagnostics is None:
                raise
            diagnostics.append(
                ConfigurationDiagnostic(
                    mcp_name=mcp_name,
                    reason=_single_line_safe_text(error.error.message),
                )
            )
    return MappingProxyType(parsed)


def _parse_configuration(
    document: dict[str, object],
    *,
    diagnostics: list[ConfigurationDiagnosticValue] | None = None,
) -> UserConfiguration:
    return UserConfiguration(
        runtime=_parse_runtime(document, diagnostics=diagnostics),
        memory=_parse_memory(document, diagnostics=diagnostics),
        models=_parse_models(document, diagnostics=diagnostics),
        mcp=_parse_mcp(document, diagnostics=diagnostics),
    )


def _configuration_revision(content: bytes) -> str:
    return f"sha256:{sha256(content).hexdigest()}"


def _editable_configuration_fields(
    configuration: UserConfiguration,
) -> Mapping[str, Mapping[str, object]]:
    providers = {
        provider_id: {
            "protocol": provider.protocol,
            "base_url": provider.base_url,
            "models": provider.models,
            "api_key": {"configured": bool(provider.api_key)},
        }
        for provider_id, provider in configuration.models.providers.items()
    }
    routes = {
        route_name: {
            "provider_id": route.provider_id,
            "model": route.model,
            "context_window": route.context_window,
            "max_output": route.max_output,
            "temperature": route.temperature,
            "reasoning_effort": route.reasoning_effort,
            "timeout": route.timeout,
        }
        for route_name, route in configuration.models.routes.items()
    }
    mcp = {
        server_name: _editable_mcp_server_fields(server)
        for server_name, server in configuration.mcp.items()
    }
    return {
        "runtime": {
            "max_tool_result_chars": configuration.runtime.max_tool_result_chars,
            "max_iterations": configuration.runtime.max_iterations,
            "enable_skill_always_load": configuration.runtime.enable_skill_always_load,
            "compact_ratio": configuration.runtime.compact_ratio,
            "permission_level": configuration.runtime.permission_level,
            "exec_shell": configuration.runtime.exec_shell,
        },
        "memory": {
            "batch_size": configuration.memory.batch_size,
            "schedule": configuration.memory.schedule,
        },
        "models": {"providers": providers, "routes": routes},
        "mcp": mcp,
    }


def _editable_mcp_server_fields(server: MCPServerConfiguration) -> dict[str, object]:
    return {
        "enabled": server.enabled,
        "transport": server.transport,
        "command": server.command,
        "args": server.args,
        "cwd": None if server.cwd is None else str(server.cwd),
        "url": server.url,
        "headers": {name: {"configured": bool(value)} for name, value in server.headers.items()},
        "connect_timeout": server.connect_timeout,
        "call_timeout": server.call_timeout,
        "tool_keywords": {
            tool_name: keywords for tool_name, keywords in server.tool_keywords.items()
        },
    }


def _editable_field_value(section: str, field: str, value: object) -> object:
    name = f"{section}.{field}"
    if section == "runtime":
        if field == "max_tool_result_chars":
            if _parse_default_integer(value, 1000, 1_000_000) is None:
                raise ConfigFieldError(name, "must be an integer from 1000 to 1000000")
            return value
        if field == "max_iterations":
            if _parse_default_integer(value, 50, None) is None:
                raise ConfigFieldError(name, "must be an integer at least 50")
            return value
        if field == "enable_skill_always_load":
            if _parse_default_boolean(value) is None:
                raise ConfigFieldError(name, "must be a boolean")
            return value
        if field == "compact_ratio":
            parsed = _parse_default_compact_ratio(value)
            if parsed is None:
                raise ConfigFieldError(name, "must be a number from 0.5 to 0.95")
            return parsed
        if field == "permission_level":
            if _parse_default_permission_level(value) is None:
                raise ConfigFieldError(name, "must be read-only, workspace-write, or full-access")
            return value
        if field == "exec_shell":
            if _parse_default_exec_shell(value) is None:
                raise ConfigFieldError(name, "must be auto, powershell, or pwsh")
            return value
    elif section == "memory":
        if field == "batch_size":
            if _parse_default_integer(value, 1, 1000) is None:
                raise ConfigFieldError(name, "must be an integer from 1 to 1000")
            return value
        if field == "schedule":
            if _parse_default_schedule(value) is None:
                raise ConfigFieldError(name, "must be a valid five-field cron expression")
            return value
    _invalid("config.fields", "contains a field that is not editable")


def _editable_table(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _invalid(field, "must be a table")
    table = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) for key in table):
        _invalid(field, "must contain string field names")
    return cast(Mapping[str, object], table)


def _reject_unknown_fields(table: Mapping[str, object], allowed: set[str], field: str) -> None:
    unknown = next((name for name in table if name not in allowed), None)
    if unknown is not None:
        _invalid(f"{field}.{unknown}", "is not editable")


def _require_editable_value(table: Mapping[str, object], name: str, field: str) -> object:
    if name not in table:
        _invalid(f"{field}.{name}", "is required")
    return table[name]


def _editable_string_array(value: object, field: str) -> list[str]:
    parsed = _parse_string_array(value, field)
    if len(set(parsed)) != len(parsed):
        _invalid(field, "must contain unique string values")
    return list(parsed)


def _validate_provider_fields(provider_id: str, value: object) -> dict[str, object]:
    field = f"models.providers.{provider_id}"
    if not _PROVIDER_ID_PATTERN.fullmatch(provider_id):
        _invalid(field, "must use a lowercase kebab-case provider ID")
    table = _editable_table(value, field)
    _reject_unknown_fields(table, {"protocol", "base_url", "models"}, field)
    normalized: dict[str, object] = {}
    if "protocol" in table:
        protocol = _string(table["protocol"], f"{field}.protocol")
        if protocol not in {"anthropic", "openai-compatible"}:
            _invalid(f"{field}.protocol", "must be anthropic or openai-compatible")
        normalized["protocol"] = protocol
    if "base_url" in table:
        base_url = _string(table["base_url"], f"{field}.base_url")
        if not _has_absolute_http_url(base_url):
            _invalid(f"{field}.base_url", "must be an absolute HTTP or HTTPS URL")
        normalized["base_url"] = base_url
    if "models" in table:
        normalized["models"] = _editable_string_array(table["models"], f"{field}.models")
    return normalized


def _validate_route_fields(route_name: str, value: object) -> dict[str, object]:
    field = f"models.routes.{route_name}"
    if route_name not in _ROUTE_NAMES:
        _invalid(field, "is not a supported Model Route")
    table = _editable_table(value, field)
    allowed = {
        "provider_id",
        "model",
        "context_window",
        "max_output",
        "temperature",
        "reasoning_effort",
        "timeout",
    }
    _reject_unknown_fields(table, allowed, field)
    normalized: dict[str, object] = {}
    if "provider_id" in table:
        provider_id = _string(table["provider_id"], f"{field}.provider_id", nonempty=True)
        if not _PROVIDER_ID_PATTERN.fullmatch(provider_id):
            _invalid(f"{field}.provider_id", "must be a lowercase kebab-case provider ID")
        normalized["provider_id"] = provider_id
    if "model" in table:
        normalized["model"] = _string(table["model"], f"{field}.model", nonempty=True)
    if "context_window" in table:
        normalized["context_window"] = _integer(
            table["context_window"], f"{field}.context_window", 1024, 10_000_000
        )
    if "max_output" in table:
        normalized["max_output"] = _integer(
            table["max_output"], f"{field}.max_output", 1, 9_999_999
        )
    if "context_window" in normalized and "max_output" in normalized:
        if cast(int, normalized["max_output"]) >= cast(int, normalized["context_window"]):
            _invalid(f"{field}.max_output", "must be less than context_window")
    if "temperature" in table:
        normalized["temperature"] = _number(table["temperature"], f"{field}.temperature", 0, 2)
    if "reasoning_effort" in table:
        reasoning_effort = _string(table["reasoning_effort"], f"{field}.reasoning_effort")
        if _parse_default_reasoning_effort(reasoning_effort) is None:
            _invalid(f"{field}.reasoning_effort", "must be low, medium, high, xhigh, or max")
        normalized["reasoning_effort"] = reasoning_effort
    if "timeout" in table:
        normalized["timeout"] = _integer(table["timeout"], f"{field}.timeout", 1, 600)
    return normalized


def _validate_redacted_headers(value: object, field: str) -> dict[str, dict[str, bool]]:
    table = _editable_table(value, field)
    headers: dict[str, dict[str, bool]] = {}
    for header_name, header_value in table.items():
        if not header_name or header_name != header_name.strip():
            _invalid(field, "must contain nonempty header names without surrounding whitespace")
        redacted = _editable_table(header_value, f"{field}.{header_name}")
        _reject_unknown_fields(redacted, {"configured"}, f"{field}.{header_name}")
        configured = _require_editable_value(redacted, "configured", f"{field}.{header_name}")
        if not isinstance(configured, bool):
            _invalid(f"{field}.{header_name}.configured", "must be a boolean")
        headers[header_name] = {"configured": configured}
    return headers


def _validate_mcp_fields(mcp_name: str, value: object) -> dict[str, object]:
    field = f"mcp.{mcp_name}"
    if _MCP_NAME_PATTERN.fullmatch(mcp_name) is None:
        _invalid(field, "must use a lowercase name with up to 64 letters, digits, '_' or '-'")
    table = _editable_table(value, field)
    allowed = {
        "enabled",
        "transport",
        "command",
        "args",
        "cwd",
        "url",
        "headers",
        "connect_timeout",
        "call_timeout",
        "tool_keywords",
    }
    _reject_unknown_fields(table, allowed, field)
    normalized: dict[str, object] = {}
    transport: str | None = None
    if "transport" in table:
        transport = _string(table["transport"], f"{field}.transport")
        if transport not in _MCP_TRANSPORTS:
            _invalid(f"{field}.transport", "must be either 'stdio' or 'streamable-http'")
        normalized["transport"] = transport
    if "enabled" in table:
        normalized["enabled"] = _boolean(table["enabled"], f"{field}.enabled")
    for timeout_name in ("connect_timeout", "call_timeout"):
        if timeout_name in table:
            normalized[timeout_name] = _integer(
                table[timeout_name], f"{field}.{timeout_name}", 1, _MCP_MAX_TIMEOUT
            )
    if "tool_keywords" in table:
        keywords_table = _editable_table(table["tool_keywords"], f"{field}.tool_keywords")
        keywords: dict[str, list[str]] = {}
        for remote_name, raw_keywords in keywords_table.items():
            if not remote_name:
                _invalid(f"{field}.tool_keywords", "must contain nonempty remote Tool names")
            try:
                keywords[remote_name] = list(normalize_mcp_tool_keywords(raw_keywords))
            except TypeError:
                _invalid(f"{field}.tool_keywords.{remote_name}", "must be an array of strings")
            except ValueError:
                _invalid(f"{field}.tool_keywords.{remote_name}", "must contain English terms")
        normalized["tool_keywords"] = keywords
    if "headers" in table:
        normalized["headers"] = _validate_redacted_headers(table["headers"], f"{field}.headers")
    if "command" in table:
        command = table["command"]
        normalized["command"] = (
            None if command is None else _string(command, f"{field}.command", nonempty=True)
        )
    if "args" in table:
        normalized["args"] = list(_parse_string_array(table["args"], f"{field}.args"))
    if "cwd" in table:
        cwd = table["cwd"]
        normalized["cwd"] = None if cwd is None else _string(cwd, f"{field}.cwd", nonempty=True)
    if "url" in table:
        url = table["url"]
        normalized["url"] = None if url is None else _string(url, f"{field}.url", nonempty=True)
        if url is not None and not _has_absolute_http_url(cast(str, normalized["url"])):
            _invalid(f"{field}.url", "must be an absolute HTTP or HTTPS URL")
    if transport == "stdio":
        if normalized.get("url") is not None:
            _invalid(f"{field}.url", "must be null for the stdio transport")
        if normalized.get("headers"):
            _invalid(f"{field}.headers", "must be empty for the stdio transport")
    elif transport == "streamable-http":
        if normalized.get("command") is not None:
            _invalid(f"{field}.command", "must be null for the streamable-http transport")
        if normalized.get("args") not in (None, []):
            _invalid(f"{field}.args", "must be empty for the streamable-http transport")
        if normalized.get("cwd") is not None:
            _invalid(f"{field}.cwd", "must be null for the streamable-http transport")
    return normalized


def _validate_editable_fields(fields: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if not isinstance(fields, Mapping):
        _invalid("config.fields", "must be a table")
    normalized: dict[str, dict[str, object]] = {}
    for section, raw_values in fields.items():
        if section not in {"runtime", "memory", "models", "mcp"}:
            _invalid("config.fields", "contains a section that is not editable")
        if not isinstance(raw_values, Mapping):
            _invalid(f"config.fields.{section}", "must be a table")
        if section == "models":
            model_values = _editable_table(raw_values, "config.fields.models")
            _reject_unknown_fields(model_values, {"providers", "routes"}, "config.fields.models")
            models: dict[str, object] = {}
            if "providers" in model_values:
                providers = _editable_table(
                    model_values["providers"], "config.fields.models.providers"
                )
                models["providers"] = {
                    provider_id: _validate_provider_fields(provider_id, provider)
                    for provider_id, provider in providers.items()
                }
            if "routes" in model_values:
                routes = _editable_table(model_values["routes"], "config.fields.models.routes")
                models["routes"] = {
                    route_name: _validate_route_fields(route_name, route)
                    for route_name, route in routes.items()
                }
            normalized[section] = models
            continue
        if section == "mcp":
            servers = _editable_table(raw_values, "config.fields.mcp")
            normalized[section] = {
                server_name: _validate_mcp_fields(server_name, server)
                for server_name, server in servers.items()
            }
            continue
        section_values: dict[str, object] = {}
        for field_name, value in raw_values.items():
            if not isinstance(field_name, str):
                _invalid(f"config.fields.{section}", "contains an invalid field name")
            section_values[field_name] = _editable_field_value(section, field_name, value)
        normalized[section] = section_values
    return normalized


def _mutable_toml_table(
    parent: MutableMapping[str, object], key: str, field: str
) -> MutableMapping[str, object]:
    value = parent.get(key)
    if value is None:
        value = tomlkit.table()
        parent[key] = value
    if not isinstance(value, MutableMapping):
        _invalid(field, "must be a table")
    return cast(MutableMapping[str, object], value)


def _set_toml_table_value(table: MutableMapping[str, object], key: str, value: object) -> None:
    if value is None:
        table.pop(key, None)
    elif isinstance(value, list):
        table[key] = list(value)
    elif isinstance(value, Mapping):
        nested = tomlkit.table()
        for nested_key, nested_value in value.items():
            _set_toml_table_value(nested, nested_key, nested_value)
        table[key] = nested
    else:
        table[key] = value


def _apply_model_fields(
    document: MutableMapping[str, object], values: Mapping[str, object]
) -> None:
    models = _mutable_toml_table(document, "models", "models")
    if "providers" in values:
        providers = _mutable_toml_table(models, "providers", "models.providers")
        provider_values = cast(Mapping[str, Mapping[str, object]], values["providers"])
        for provider_id in tuple(providers):
            if provider_id not in provider_values:
                del providers[provider_id]
        for provider_id, provider in provider_values.items():
            table = _mutable_toml_table(providers, provider_id, f"models.providers.{provider_id}")
            if "api_key" not in table:
                table["api_key"] = ""
            for field, value in provider.items():
                _set_toml_table_value(table, field, value)
    if "routes" in values:
        routes = _mutable_toml_table(models, "routes", "models.routes")
        route_values = cast(Mapping[str, Mapping[str, object]], values["routes"])
        for route_name in tuple(routes):
            if route_name in _ROUTE_NAMES and route_name not in route_values:
                del routes[route_name]
        for route_name, route in route_values.items():
            table = _mutable_toml_table(routes, route_name, f"models.routes.{route_name}")
            for field, value in route.items():
                _set_toml_table_value(table, field, value)


def _apply_mcp_fields(document: MutableMapping[str, object], values: Mapping[str, object]) -> None:
    mcp = _mutable_toml_table(document, "mcp", "mcp")
    servers = _mutable_toml_table(mcp, "servers", "mcp.servers")
    server_values = cast(Mapping[str, Mapping[str, object]], values)
    for server_name in tuple(servers):
        if server_name not in server_values:
            del servers[server_name]
    for server_name, server in server_values.items():
        table = _mutable_toml_table(servers, server_name, f"mcp.{server_name}")
        effective_transport = server.get("transport", table.get("transport"))
        if isinstance(effective_transport, str) and effective_transport in _MCP_TRANSPORTS:
            _validate_mcp_fields(server_name, {"transport": effective_transport, **server})
        for field_name in ("enabled", "transport", "connect_timeout", "call_timeout"):
            if field_name in server:
                _set_toml_table_value(table, field_name, server[field_name])
        transport = cast(str | None, server.get("transport", table.get("transport")))
        if transport == "stdio":
            for field in ("command", "args", "cwd"):
                if field in server:
                    _set_toml_table_value(table, field, server[field])
            for field in ("url", "headers"):
                if field in table:
                    del table[field]
        else:
            if "url" in server:
                _set_toml_table_value(table, "url", server["url"])
            for field in ("command", "args", "cwd"):
                if field in table:
                    del table[field]
            if "headers" in server:
                header_values = cast(Mapping[str, object], server["headers"])
                headers = _mutable_toml_table(table, "headers", f"mcp.{server_name}.headers")
                for header_name in tuple(headers):
                    if header_name not in header_values:
                        del headers[header_name]
                for header_name in header_values:
                    if header_name not in headers:
                        headers[header_name] = ""
                if not headers:
                    del table["headers"]
        if "tool_keywords" in server:
            keyword_values = cast(Mapping[str, object], server["tool_keywords"])
            keywords = _mutable_toml_table(
                table, "tool_keywords", f"mcp.{server_name}.tool_keywords"
            )
            for remote_name in tuple(keywords):
                if remote_name not in keyword_values:
                    del keywords[remote_name]
            for remote_name, raw_keywords in keyword_values.items():
                _set_toml_table_value(keywords, remote_name, raw_keywords)
            if not keywords and "tool_keywords" in table:
                del table["tool_keywords"]


def _apply_secret_changes(
    document: MutableMapping[str, object], changes: Mapping[str, object]
) -> None:
    if not isinstance(changes, Mapping):
        _invalid("config.secrets", "must be a table")
    models = document.get("models")
    mcp = document.get("mcp")
    for path, raw_change in changes.items():
        if not isinstance(path, str):
            _invalid("config.secrets", "must contain string paths")
        change = _editable_table(raw_change, f"config.secrets.{path}")
        _reject_unknown_fields(change, {"action", "value"}, f"config.secrets.{path}")
        action = _string(
            _require_editable_value(change, "action", f"config.secrets.{path}"),
            f"config.secrets.{path}.action",
        )
        if action not in {"keep", "replace", "clear"}:
            _invalid(f"config.secrets.{path}.action", "must be keep, replace, or clear")
        if action == "replace":
            value = _string(
                _require_editable_value(change, "value", f"config.secrets.{path}"),
                f"config.secrets.{path}.value",
            )
            if not value:
                _invalid(f"config.secrets.{path}.value", "must be a nonempty string")
        elif "value" in change:
            _invalid(f"config.secrets.{path}.value", "is only valid for replace")
        if path.startswith("models.providers.") and path.endswith(".api_key"):
            provider_id = path[len("models.providers.") : -len(".api_key")]
            if not isinstance(models, MutableMapping):
                _invalid(path, "does not identify an existing Provider")
            providers = models.get("providers")
            if not isinstance(providers, MutableMapping) or provider_id not in providers:
                _invalid(path, "does not identify an existing Provider")
            provider = providers[provider_id]
            if not isinstance(provider, MutableMapping):
                _invalid(path, "does not identify an editable Provider")
            if action == "replace":
                provider["api_key"] = value
            elif action == "clear":
                provider["api_key"] = ""
            continue
        if path.startswith("mcp.") and ".headers." in path:
            server_name, header_name = path[4:].split(".headers.", 1)
            if not server_name or not header_name or header_name != header_name.strip():
                _invalid(path, "does not identify an editable MCP header")
            if not isinstance(mcp, MutableMapping):
                _invalid(path, "does not identify an existing MCP Server")
            servers = mcp.get("servers")
            if not isinstance(servers, MutableMapping) or server_name not in servers:
                _invalid(path, "does not identify an existing MCP Server")
            server = servers[server_name]
            if (
                not isinstance(server, MutableMapping)
                or server.get("transport") != "streamable-http"
            ):
                _invalid(path, "does not identify an editable MCP header")
            headers = server.get("headers")
            if headers is None:
                headers = tomlkit.table()
                server["headers"] = headers
            if not isinstance(headers, MutableMapping):
                _invalid(path, "does not identify an editable MCP header")
            if action == "replace":
                headers[header_name] = value
            elif action == "clear" and header_name in headers:
                del headers[header_name]
            continue
        _invalid(path, "is not an editable secret")


def _require_complete_candidate(
    configuration: UserConfiguration,
    diagnostics: tuple[ConfigurationDiagnosticValue, ...] | list[ConfigurationDiagnosticValue],
) -> None:
    if "default" not in configuration.models.routes:
        raise _missing_default_route_error()
    for provider_id, configured_provider in configuration.models.providers.items():
        if configured_provider.protocol not in {"anthropic", "openai-compatible"}:
            _invalid(
                f"models.providers.{provider_id}.protocol", "must be anthropic or openai-compatible"
            )
        if not _has_absolute_http_url(configured_provider.base_url):
            _invalid(
                f"models.providers.{provider_id}.base_url", "must be an absolute HTTP or HTTPS URL"
            )
    for route_name, route in configuration.models.routes.items():
        provider = configuration.models.providers.get(route.provider_id)
        if provider is None:
            _invalid(
                f"models.routes.{route_name}.provider_id",
                "must reference an existing Model Provider",
            )
        if route.model not in provider.models:
            _invalid(f"models.routes.{route_name}.model", "must reference an available model")
        if _usable_route(configuration.models, route_name) is None:
            _invalid(f"models.routes.{route_name}", "must reference a usable Model Provider")
    if diagnostics:
        raise ConfigError(
            ErrorInfo("config_invalid", "The complete User Configuration contains invalid fields.")
        )


def _safe_projection_row[T](
    value: object,
    defaults: Mapping[str, object],
    prefix: str,
    parse: Callable[[Mapping[str, object]], T],
) -> T:
    """Retain valid siblings while replacing only unprojectable known field values."""
    row = dict(defaults)
    if isinstance(value, Mapping):
        row.update({name: item for name, item in value.items() if name in defaults})
    while True:
        try:
            return parse(row)
        except ConfigError as error:
            field_name = next(iter(error.field_errors), "")
            relative = field_name.removeprefix(prefix + ".").replace("mcp.servers.", "mcp.", 1)
            name, _, nested = relative.partition(".")
            if name not in row:
                raise
            field_value = row[name]
            if nested and isinstance(field_value, Mapping):
                values = dict(field_value)
                values.pop(nested, None)
                row[name] = values
            elif row[name] != defaults[name]:
                row[name] = defaults[name]
            else:
                raise


def _safe_web_configuration(document: Mapping[str, object]) -> tuple[UserConfiguration, bool]:
    """Project a parseable document without allowing malformed values into the Web form."""
    default_document = _table(tomllib.loads(DEFAULT_CONFIG_TEMPLATE), "configuration")
    safe_document: dict[str, object] = {}
    has_issues = False

    default_runtime = _table(default_document.get("runtime", {}), "runtime")
    runtime_value = document.get("runtime", {})
    runtime = dict(default_runtime)
    if not isinstance(runtime_value, Mapping):
        has_issues = True
    else:
        for field_name in default_runtime:
            if field_name not in runtime_value:
                continue
            try:
                _parse_runtime({"runtime": {field_name: runtime_value[field_name]}}, diagnostics=[])
            except ConfigError:
                has_issues = True
            else:
                runtime[field_name] = runtime_value[field_name]
    safe_document["runtime"] = runtime

    default_memory = _table(default_document.get("memory", {}), "memory")
    memory_value = document.get("memory", {})
    memory = dict(default_memory)
    if not isinstance(memory_value, Mapping):
        has_issues = True
    else:
        for field_name in default_memory:
            if field_name not in memory_value:
                continue
            try:
                _parse_memory({"memory": {field_name: memory_value[field_name]}}, diagnostics=[])
            except ConfigError:
                has_issues = True
            else:
                memory[field_name] = memory_value[field_name]
    safe_document["memory"] = memory

    default_models = _table(default_document.get("models", {}), "models")
    default_providers = _table(default_models.get("providers", {}), "models.providers")
    default_routes = _table(default_models.get("routes", {}), "models.routes")
    models_value = document.get("models", {})
    raw_providers: Mapping[object, object] = {}
    raw_routes: Mapping[object, object] = {}
    if not isinstance(models_value, Mapping):
        has_issues = True
    else:
        providers_value = models_value.get("providers", {})
        routes_value = models_value.get("routes", {})
        if not isinstance(providers_value, Mapping):
            has_issues = True
        else:
            raw_providers = providers_value
        if not isinstance(routes_value, Mapping):
            has_issues = True
        else:
            raw_routes = routes_value

    use_model_defaults = "models" not in document or not isinstance(models_value, Mapping)
    providers: dict[str, object] = (
        {
            str(provider_id): dict(provider)
            for provider_id, provider in default_providers.items()
            if isinstance(provider_id, str) and isinstance(provider, Mapping)
        }
        if use_model_defaults
        else {}
    )
    unsafe_providers: dict[str, ProviderConfiguration] = {}
    for provider_id, raw_provider in raw_providers.items():
        if not isinstance(provider_id, str):
            has_issues = True
            continue
        parse_id = provider_id if _PROVIDER_ID_PATTERN.fullmatch(provider_id) else "repair"
        if not isinstance(raw_provider, Mapping):
            has_issues = True
        provider = _safe_projection_row(
            raw_provider,
            {"protocol": "openai-compatible", "base_url": "", "api_key": "", "models": []},
            f"models.providers.{parse_id}",
            partial(_parse_provider, parse_id),
        )
        if parse_id != provider_id:
            unsafe_providers[provider_id] = replace(provider, provider_id=provider_id)
            continue
        providers[provider_id] = {
            "protocol": provider.protocol,
            "base_url": provider.base_url,
            "api_key": provider.api_key,
            "models": list(provider.models),
        }

    routes: dict[str, object] = (
        {
            str(route_name): dict(route)
            for route_name, route in default_routes.items()
            if isinstance(route_name, str) and isinstance(route, Mapping)
        }
        if use_model_defaults
        else {}
    )
    for route_name, raw_route in raw_routes.items():
        if not isinstance(route_name, str) or route_name not in _ROUTE_NAMES:
            continue
        if not isinstance(raw_route, Mapping):
            has_issues = True
        default_route = dict(cast(Mapping[str, object], default_routes["default"]))
        default_route["max_output"] = 1
        default_route["reasoning_effort"] = _DEFAULT_REASONING_EFFORT
        route = _safe_projection_row(
            raw_route,
            default_route,
            f"models.routes.{route_name}",
            partial(_parse_route, route_name, diagnostics=[]),
        )
        routes[route_name] = {
            "provider_id": route.provider_id,
            "model": route.model,
            "context_window": route.context_window,
            "max_output": route.max_output,
            "temperature": route.temperature,
            "reasoning_effort": route.reasoning_effort,
            "timeout": route.timeout,
        }

    mcp: dict[str, object] = {}
    unsafe_mcp: dict[str, MCPServerConfiguration] = {}
    mcp_value = document.get("mcp", {})
    if not isinstance(mcp_value, Mapping):
        has_issues = True
    else:
        servers_value = mcp_value.get("servers", {})
        if not isinstance(servers_value, Mapping):
            has_issues = True
        else:
            for server_name, server in servers_value.items():
                if not isinstance(server_name, str):
                    has_issues = True
                    continue
                parse_name = server_name if _MCP_NAME_PATTERN.fullmatch(server_name) else "repair"
                if not isinstance(server, Mapping):
                    server = {}
                transport = server.get("transport", "stdio")
                defaults: dict[str, object] = {
                    "enabled": False,
                    "transport": (
                        transport
                        if isinstance(transport, str) and transport in _MCP_TRANSPORTS
                        else "stdio"
                    ),
                    "connect_timeout": _MCP_DEFAULT_CONNECT_TIMEOUT,
                    "call_timeout": _MCP_DEFAULT_CALL_TIMEOUT,
                    "tool_keywords": {},
                }
                if defaults["transport"] == "streamable-http":
                    defaults.update(url="http://127.0.0.1", headers={})
                else:
                    defaults.update(command="python", args=[])
                    if "cwd" in server:
                        defaults["cwd"] = None
                projected = _safe_projection_row(
                    server,
                    defaults,
                    f"mcp.{parse_name}",
                    partial(_parse_mcp_server, parse_name),
                )
                if parse_name != server_name:
                    unsafe_mcp[server_name] = replace(projected, mcp_name=server_name)
                    continue
                safe_server = {name: value for name, value in defaults.items()}
                safe_server.update(
                    enabled=projected.enabled,
                    transport=projected.transport,
                    connect_timeout=projected.connect_timeout,
                    call_timeout=projected.call_timeout,
                    tool_keywords={
                        name: list(value) for name, value in projected.tool_keywords.items()
                    },
                )
                if projected.transport == "streamable-http":
                    safe_server.update(url=projected.url, headers=dict(projected.headers))
                else:
                    safe_server.update(command=projected.command, args=list(projected.args))
                    if projected.cwd is not None:
                        safe_server["cwd"] = str(projected.cwd)
                mcp[server_name] = safe_server

    safe_document["models"] = {"providers": providers, "routes": routes}
    safe_document["mcp"] = {"servers": mcp}
    diagnostics: list[ConfigurationDiagnosticValue] = []
    configuration = _parse_configuration(safe_document, diagnostics=diagnostics)
    if unsafe_providers or unsafe_mcp:
        configuration = replace(
            configuration,
            models=replace(
                configuration.models,
                providers=MappingProxyType({**configuration.models.providers, **unsafe_providers}),
            ),
            mcp=MappingProxyType({**configuration.mcp, **unsafe_mcp}),
        )
    return configuration, has_issues or bool(diagnostics)


def _config_web_error(code: str, message: str) -> Mapping[str, str]:
    return {"code": code, "message": message}


def _create_private_backup(target: Path, content: bytes) -> bool:
    """Protect an empty temporary file before writing and publishing exact bytes."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=".config-backup-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        HOST_FILESYSTEM.protect_private_file(temporary)
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            if stream.write(content) != len(content):
                raise OSError("Malformed configuration backup was not fully written")
            stream.flush()
            HOST_FILESYSTEM.sync_file(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            return False
        HOST_FILESYSTEM.sync_parent_directory(target.parent)
        return True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _prepare_repair_tables(
    document: MutableMapping[str, object], normalized: Mapping[str, Mapping[str, object]]
) -> None:
    """Replace touched invalid table shapes while preserving parseable sibling tables."""
    for section, values in normalized.items():
        if section in document and not isinstance(document[section], MutableMapping):
            document[section] = tomlkit.table()
        if section == "models":
            models = _mutable_toml_table(document, section, section)
            for collection_name in values:
                if collection_name in models and not isinstance(
                    models[collection_name], MutableMapping
                ):
                    models[collection_name] = tomlkit.table()
                collection = _mutable_toml_table(
                    models, collection_name, f"models.{collection_name}"
                )
                rows = cast(Mapping[str, object], values[collection_name])
                for row_name in rows:
                    if row_name in collection and not isinstance(
                        collection[row_name], MutableMapping
                    ):
                        collection[row_name] = tomlkit.table()
        elif section == "mcp":
            mcp = _mutable_toml_table(document, section, section)
            if "servers" in mcp and not isinstance(mcp["servers"], MutableMapping):
                mcp["servers"] = tomlkit.table()
            servers = _mutable_toml_table(mcp, "servers", "mcp.servers")
            for row_name in values:
                if row_name in servers and not isinstance(servers[row_name], MutableMapping):
                    servers[row_name] = tomlkit.table()
                if isinstance(servers.get(row_name), MutableMapping):
                    row = cast(MutableMapping[str, object], servers[row_name])
                    changed = cast(Mapping[str, object], values[row_name])
                    for nested in ("headers", "tool_keywords"):
                        if (
                            nested in changed
                            and nested in row
                            and not isinstance(row[nested], MutableMapping)
                        ):
                            row[nested] = tomlkit.table()


def _require_explicit_transport_repair(
    document: MutableMapping[str, object], fields: Mapping[str, object]
) -> None:
    mcp = document.get("mcp")
    if not isinstance(mcp, Mapping) or not isinstance(mcp.get("servers"), Mapping):
        return
    servers = cast(Mapping[str, object], mcp["servers"])
    for name, changes in fields.items():
        original = servers.get(name)
        if not isinstance(original, Mapping) or not isinstance(changes, Mapping):
            continue
        transport = original.get("transport")
        if changes.get("transport", transport) != transport:
            continue
        incompatible = (
            ("url", "headers")
            if transport == "stdio"
            else ("command", "args", "cwd")
            if transport == "streamable-http"
            else ()
        )
        for field_name in incompatible:
            if field_name in original:
                _invalid(
                    f"mcp.{name}.{field_name}",
                    "requires an explicit transport change or Server removal before repair",
                )


class ConfigLoader:
    """Access User Configuration beneath an injected fixed Agent Home."""

    def __init__(self, agent_home: AgentHome) -> None:
        self.agent_home = agent_home
        self._diagnostics: tuple[ConfigurationDiagnosticValue, ...] = ()

    @property
    def path(self) -> Path:
        return self.agent_home.path / "config.toml"

    @property
    def diagnostics(self) -> tuple[ConfigurationDiagnosticValue, ...]:
        """Return diagnostics from the most recent successful configuration parse."""
        return self._diagnostics

    def ensure_default(self) -> bool:
        """Create the accepted default template when missing."""
        self._diagnostics = ()
        self.agent_home.initialize()
        return HOST_FILESYSTEM.atomic_create_text(self.path, DEFAULT_CONFIG_TEMPLATE)

    @staticmethod
    def revision_from_bytes(content: bytes) -> str:
        """Return the opaque revision used by compare-and-swap edits."""
        return _configuration_revision(content)

    def revision(self) -> str:
        """Return the current raw-file revision without exposing its contents."""
        return _configuration_revision(self.path.read_bytes())

    def editable_snapshot(self) -> ConfigEditableSnapshot:
        """Read the safe structured configuration projection for a Web editor."""
        try:
            content = self.path.read_bytes()
            document = _table(tomllib.loads(content.decode("utf-8")), "configuration")
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            if isinstance(error, OSError):
                raise
            raise ConfigError(
                ErrorInfo(
                    "config_parse_error",
                    "User Configuration TOML could not be parsed.",
                )
            ) from error
        diagnostics: list[ConfigurationDiagnosticValue] = []
        configuration = _parse_configuration(document, diagnostics=diagnostics)
        self._diagnostics = tuple(diagnostics)
        return ConfigEditableSnapshot(
            revision=_configuration_revision(content),
            fields=_editable_configuration_fields(configuration),
            configuration=configuration,
            diagnostics=tuple(diagnostics),
        )

    def web_snapshot(self) -> ConfigWebSnapshot:
        """Return a redacted projection that remains available during first-use repair."""
        try:
            content = self.path.read_bytes()
        except FileNotFoundError:
            configuration, _ = _safe_web_configuration({})
            self._diagnostics = ()
            return ConfigWebSnapshot(
                revision=_configuration_revision(b""),
                fields=_editable_configuration_fields(configuration),
                configuration=configuration,
                state="missing",
                repair_required=True,
                backup_required=False,
                requires_secret_reentry=True,
                error=_config_web_error(
                    "config_missing",
                    "A User Configuration is required before MyClaw can run.",
                ),
            )
        except OSError:
            raise

        revision = _configuration_revision(content)
        try:
            document = _table(tomllib.loads(content.decode("utf-8")), "configuration")
        except (UnicodeDecodeError, tomllib.TOMLDecodeError):
            configuration, _ = _safe_web_configuration({})
            self._diagnostics = ()
            return ConfigWebSnapshot(
                revision=revision,
                fields=_editable_configuration_fields(configuration),
                configuration=configuration,
                state="malformed",
                repair_required=True,
                backup_required=True,
                requires_secret_reentry=True,
                error=_config_web_error(
                    "config_parse_error",
                    "User Configuration TOML could not be parsed.",
                ),
            )

        try:
            diagnostics: list[ConfigurationDiagnosticValue] = []
            configuration = _parse_configuration(document, diagnostics=diagnostics)
            _require_complete_candidate(configuration, diagnostics)
        except ConfigError:
            configuration, _ = _safe_web_configuration(document)
            valid = False
        else:
            valid = True
        self._diagnostics = ()
        if not valid:
            return ConfigWebSnapshot(
                revision=revision,
                fields=_editable_configuration_fields(configuration),
                configuration=configuration,
                state="invalid",
                repair_required=True,
                backup_required=False,
                requires_secret_reentry=not any(
                    provider.api_key.strip() for provider in configuration.models.providers.values()
                ),
                error=_config_web_error(
                    "config_invalid",
                    "The saved User Configuration contains invalid fields.",
                ),
            )
        return ConfigWebSnapshot(
            revision=revision,
            fields=_editable_configuration_fields(configuration),
            configuration=configuration,
            state="active",
            repair_required=False,
            backup_required=False,
            requires_secret_reentry=False,
            error=None,
        )

    def patch_editable_fields(
        self,
        expected_revision: str,
        fields: Mapping[str, object],
        secrets: Mapping[str, object] | None = None,
    ) -> ConfigEditResult:
        """Atomically apply safe fields when the caller still has the latest revision."""
        if not isinstance(expected_revision, str) or not expected_revision:
            _invalid("config.revision", "must be a nonempty string")
        normalized = _validate_editable_fields(fields)
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with HOST_FILESYSTEM.exclusive_lock(lock_path):
            original_content = self.path.read_bytes()
            current_revision = _configuration_revision(original_content)
            if current_revision != expected_revision:
                raise ConfigRevisionConflict(expected_revision, current_revision)
            try:
                source_document = tomlkit.parse(original_content.decode("utf-8"))
            except (tomlkit.exceptions.ParseError, UnicodeDecodeError) as error:
                raise ConfigError(
                    ErrorInfo(
                        "config_parse_error",
                        "User Configuration TOML could not be parsed.",
                    )
                ) from error

            for section, section_values in normalized.items():
                if section == "models":
                    _apply_model_fields(source_document, section_values)
                elif section == "mcp":
                    _apply_mcp_fields(source_document, section_values)
                else:
                    table = source_document.get(section)
                    if table is None:
                        table = tomlkit.table()
                        source_document[section] = table
                    if not isinstance(table, MutableMapping):
                        _invalid(section, "must be a table")
                    for field_name, value in section_values.items():
                        table[field_name] = value

            _apply_secret_changes(source_document, {} if secrets is None else secrets)

            candidate_content = tomlkit.dumps(source_document)
            try:
                candidate = tomllib.loads(candidate_content)
            except tomllib.TOMLDecodeError as error:
                raise ConfigError(
                    ErrorInfo(
                        "config_parse_error",
                        "User Configuration TOML could not be parsed.",
                    )
                ) from error
            candidate_diagnostics: list[ConfigurationDiagnosticValue] = []
            configuration = _parse_configuration(
                _table(candidate, "configuration"),
                diagnostics=candidate_diagnostics,
            )
            _require_complete_candidate(configuration, candidate_diagnostics)

            latest_content = self.path.read_bytes()
            latest_revision = _configuration_revision(latest_content)
            if latest_revision != current_revision:
                raise ConfigRevisionConflict(expected_revision, latest_revision)
            HOST_FILESYSTEM.atomic_replace_text(self.path, candidate_content)
            self._diagnostics = tuple(candidate_diagnostics)
            return ConfigEditResult(
                revision=_configuration_revision(candidate_content.encode("utf-8")),
                fields=_editable_configuration_fields(configuration),
                configuration=configuration,
            )

    def repair_editable_fields(
        self,
        expected_revision: str,
        fields: Mapping[str, object],
        secrets: Mapping[str, object] | None = None,
    ) -> ConfigEditResult:
        """Repair a missing or malformed document using a validated default structure."""
        if not isinstance(expected_revision, str) or not expected_revision:
            _invalid("config.revision", "must be a nonempty string")
        normalized = _validate_editable_fields(fields)
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with HOST_FILESYSTEM.exclusive_lock(lock_path):
            missing = False
            HOST_FILESYSTEM.require_owned_directory(
                self.agent_home.path, within=self.agent_home.path
            )
            if HOST_FILESYSTEM.entry_exists(self.path):
                HOST_FILESYSTEM.require_owned_regular_file(self.path, within=self.agent_home.path)
            try:
                original_content = self.path.read_bytes()
            except FileNotFoundError:
                missing = True
                original_content = b""
            current_revision = _configuration_revision(original_content)
            if current_revision != expected_revision:
                raise ConfigRevisionConflict(expected_revision, current_revision)

            malformed = False
            if missing:
                source_document = tomlkit.parse(DEFAULT_CONFIG_TEMPLATE)
            else:
                try:
                    source_document = tomlkit.parse(original_content.decode("utf-8"))
                except (tomlkit.exceptions.ParseError, UnicodeDecodeError):
                    malformed = True
                    source_document = tomlkit.parse(DEFAULT_CONFIG_TEMPLATE)

            if "mcp" in normalized:
                _require_explicit_transport_repair(source_document, normalized["mcp"])
            _prepare_repair_tables(source_document, normalized)
            for section, section_values in normalized.items():
                if section == "models":
                    _apply_model_fields(source_document, section_values)
                elif section == "mcp":
                    _apply_mcp_fields(source_document, section_values)
                else:
                    table = source_document.get(section)
                    if table is None:
                        table = tomlkit.table()
                        source_document[section] = table
                    if not isinstance(table, MutableMapping):
                        _invalid(section, "must be a table")
                    for field_name, value in section_values.items():
                        table[field_name] = value

            _apply_secret_changes(source_document, {} if secrets is None else secrets)
            candidate_content = tomlkit.dumps(source_document)
            try:
                candidate = tomllib.loads(candidate_content)
            except tomllib.TOMLDecodeError as error:
                raise ConfigError(
                    ErrorInfo("config_parse_error", "User Configuration TOML could not be parsed.")
                ) from error
            candidate_diagnostics: list[ConfigurationDiagnosticValue] = []
            configuration = _parse_configuration(
                _table(candidate, "configuration"),
                diagnostics=candidate_diagnostics,
            )
            _require_complete_candidate(configuration, candidate_diagnostics)

            try:
                latest_content = self.path.read_bytes()
            except FileNotFoundError:
                latest_content = b""
            latest_revision = _configuration_revision(latest_content)
            if latest_revision != current_revision:
                raise ConfigRevisionConflict(expected_revision, latest_revision)

            backup_id: str | None = None
            if malformed:
                backup_id = self._backup_malformed_content(original_content)
            if HOST_FILESYSTEM.entry_exists(self.path):
                HOST_FILESYSTEM.require_owned_regular_file(self.path, within=self.agent_home.path)
            try:
                final_content = self.path.read_bytes()
            except FileNotFoundError:
                final_content = b""
            final_revision = _configuration_revision(final_content)
            if final_revision != current_revision:
                raise ConfigRevisionConflict(expected_revision, final_revision)
            HOST_FILESYSTEM.atomic_replace_text(self.path, candidate_content)
            self._diagnostics = tuple(candidate_diagnostics)
            return ConfigEditResult(
                revision=_configuration_revision(candidate_content.encode("utf-8")),
                fields=_editable_configuration_fields(configuration),
                configuration=configuration,
                backup_id=backup_id,
            )

    def _backup_malformed_content(self, content: bytes) -> str:
        digest = sha256(content).hexdigest()
        backup_id = f"sha256:{digest}"
        target = self.agent_home.path / f"config.toml.backup.{digest}"
        for attempt in range(32):
            if HOST_FILESYSTEM.entry_exists(target):
                HOST_FILESYSTEM.require_owned_regular_file(target, within=self.agent_home.path)
                if target.read_bytes() == content:
                    HOST_FILESYSTEM.protect_private_file(target)
                    return backup_id
                target = self.agent_home.path / (
                    f"config.toml.backup.{digest}.{attempt + 1}-{uuid4().hex[:12]}"
                )
                continue
            if _create_private_backup(target, content):
                HOST_FILESYSTEM.require_owned_regular_file(target, within=self.agent_home.path)
                if target.read_bytes() != content:
                    raise OSError("Malformed configuration backup verification failed")
                return backup_id
        raise OSError("Could not publish a unique malformed configuration backup")

    def load(self) -> UserConfiguration:
        """Load User Configuration as immutable typed values."""
        self._diagnostics = ()
        try:
            loaded: object = tomllib.loads(self.path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
            raise ConfigError(
                ErrorInfo(
                    "config_parse_error",
                    "User Configuration TOML could not be parsed.",
                )
            ) from error
        document = _table(loaded, "configuration")
        diagnostics: list[ConfigurationDiagnosticValue] = []
        configuration = _parse_configuration(document, diagnostics=diagnostics)
        self._diagnostics = tuple(diagnostics)
        return configuration

    def load_for_startup(self) -> UserConfiguration:
        """Generate missing configuration or return a startup-usable configuration."""
        try:
            if self.ensure_default():
                raise ConfigError(
                    ErrorInfo(
                        "config_missing",
                        "A default User Configuration was created; edit it before starting MyClaw.",
                    )
                )
            configuration = self.load()
            if "default" not in configuration.models.routes:
                raise _missing_default_route_error()
            return configuration
        except OSError as error:
            raise ConfigError(
                ErrorInfo(
                    "persistence_error",
                    "User Configuration could not be read or written.",
                )
            ) from error

    def update_reasoning_effort(self, effort: ReasoningEffort) -> None:
        """Persist a Runtime-Lifetime Reasoning Effort in the latest configuration."""
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            _invalid(
                "models.routes.default.reasoning_effort",
                "must be low, medium, high, xhigh, or max",
            )

        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with HOST_FILESYSTEM.exclusive_lock(lock_path):
            source_document = self._read_editable_toml()

            models = source_document.get("models", {})
            if not isinstance(models, Mapping):
                _invalid("models", "must be a table")
            routes = models.get("routes", {})
            if not isinstance(routes, MutableMapping):
                _invalid("models.routes", "must be a table")
            if "default" not in routes:
                raise _missing_default_route_error()
            default = routes["default"]
            if not isinstance(default, MutableMapping):
                _invalid("models.routes.default", "must be a table")
            default["reasoning_effort"] = effort

            chat = routes.get("chat")
            if isinstance(chat, MutableMapping):
                chat["reasoning_effort"] = effort

            self._publish_editable_toml(source_document)

    def fill_mcp_tool_keywords(
        self,
        generated: Mapping[tuple[str, str], tuple[str, ...]],
    ) -> Mapping[tuple[str, str], tuple[str, ...]]:
        """Fill still-empty MCP keyword entries in the latest configuration."""
        if not isinstance(generated, Mapping):
            raise TypeError("Generated MCP keywords must be a mapping")
        assignments: dict[tuple[str, str], tuple[str, ...]] = {}
        for identity, raw_keywords in generated.items():
            if (
                not isinstance(identity, tuple)
                or len(identity) != 2
                or not all(isinstance(item, str) and item for item in identity)
            ):
                raise TypeError("Generated MCP keyword identities must name a Server and Tool")
            keywords = normalize_mcp_tool_keywords(raw_keywords)
            if keywords:
                assignments[identity] = keywords

        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with HOST_FILESYSTEM.exclusive_lock(lock_path):
            source_document = self._read_editable_toml()
            mcp = source_document.get("mcp")
            if mcp is None:
                return MappingProxyType({})
            if not isinstance(mcp, MutableMapping):
                raise TypeError("mcp must be a table")
            servers = mcp.get("servers")
            if servers is None:
                return MappingProxyType({})
            if not isinstance(servers, MutableMapping):
                raise TypeError("mcp.servers must be a table")

            effective: dict[tuple[str, str], tuple[str, ...]] = {}
            changed = False
            for (server_name, remote_name), keywords in assignments.items():
                server = servers.get(server_name)
                if server is None:
                    continue
                if not isinstance(server, MutableMapping):
                    raise TypeError(f"mcp.servers.{server_name} must be a table")
                keyword_table = server.get("tool_keywords")
                if keyword_table is None:
                    keyword_table = tomlkit.table()
                    server["tool_keywords"] = keyword_table
                    changed = True
                if not isinstance(keyword_table, MutableMapping):
                    raise TypeError(f"mcp.servers.{server_name}.tool_keywords must be a table")

                existing = keyword_table.get(remote_name)
                if existing is not None:
                    existing_keywords = normalize_mcp_tool_keywords(existing)
                    if existing_keywords:
                        effective[(server_name, remote_name)] = existing_keywords
                        continue
                keyword_table[remote_name] = list(keywords)
                effective[(server_name, remote_name)] = keywords
                changed = True

            if changed:
                self._publish_editable_toml(source_document)
            return MappingProxyType(effective)

    def _read_editable_toml(self) -> MutableMapping[str, object]:
        try:
            content = self.path.read_text(encoding="utf-8")
            source_document = tomlkit.parse(content)
        except (tomlkit.exceptions.ParseError, UnicodeDecodeError) as error:
            raise ConfigError(
                ErrorInfo(
                    "config_parse_error",
                    "User Configuration TOML could not be parsed.",
                )
            ) from error
        return cast(MutableMapping[str, object], source_document)

    def _publish_editable_toml(self, source_document: Mapping[str, object]) -> None:
        candidate_content = tomlkit.dumps(source_document)
        candidate = tomllib.loads(candidate_content)
        candidate_diagnostics: list[ConfigurationDiagnosticValue] = []
        _parse_configuration(
            _table(candidate, "configuration"),
            diagnostics=candidate_diagnostics,
        )
        HOST_FILESYSTEM.atomic_replace_text(self.path, candidate_content)
        self._diagnostics = tuple(candidate_diagnostics)

    def view(self) -> ConfigView:
        """Return complete User Configuration text with plaintext API keys redacted."""
        self._diagnostics = ()
        content = self.path.read_text(encoding="utf-8")
        try:
            loaded: object = tomllib.loads(content)
        except tomllib.TOMLDecodeError:
            return ConfigView(
                path=self.path,
                redacted_content=_redact_unparsed_content(content),
                error=ErrorInfo(
                    "config_parse_error",
                    "User Configuration TOML could not be parsed.",
                ),
            )
        document = _table(loaded, "configuration")
        error: ErrorInfo | None = None
        diagnostics: list[ConfigurationDiagnosticValue] = []
        effective_compact_ratio: float | None = None
        effective_permission_level: PermissionLevel | None = None
        effective_exec_shell: ExecShell | None = None
        try:
            configuration = _parse_configuration(document, diagnostics=diagnostics)
        except ConfigError as config_error:
            error = config_error.error
        else:
            self._diagnostics = tuple(diagnostics)
            effective_compact_ratio = configuration.runtime.compact_ratio
            effective_permission_level = configuration.runtime.permission_level
            effective_exec_shell = configuration.runtime.exec_shell
        return ConfigView(
            path=self.path,
            redacted_content=_redact_parsed_content(content),
            error=error,
            diagnostics=tuple(diagnostics),
            effective_compact_ratio=effective_compact_ratio,
            effective_permission_level=effective_permission_level,
            effective_exec_shell=effective_exec_shell,
        )
