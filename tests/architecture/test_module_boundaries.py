import ast
import importlib.util
import inspect
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from myclaw.agent.context import ContextBuilder

PROJECT_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "myclaw"
_CLI_PATH = Path("myclaw/terminal/cli.py")
_CLI_TOOL_IMPORTS = frozenset(
    {
        ("myclaw.agent.tools.mcp_runtime", "MCPRuntimeManager"),
        ("myclaw.agent.tools.mcp_runtime", "MCPServerFailure"),
        ("myclaw.agent.tools.mcp_runtime", "MCPSnapshotReport"),
        ("myclaw.agent.tools.mcp_runtime", "MCPStartupReport"),
        ("myclaw.agent.tools.mcp_runtime", "MCPToolSnapshot"),
        ("myclaw.agent.tools.mcp_keywords", "MCPKeywordPreparer"),
        ("myclaw.agent.tools.tool_gateway", "BUILT_IN_TOOL_NAMES"),
        ("myclaw.agent.tools.core.exec_host", "EXEC_CAPABILITY_ERROR"),
        ("myclaw.agent.tools.core.exec_host", "create_exec_host"),
        ("myclaw.agent.tools.core.exec_host", "resolve_exec_shell"),
    }
)
_TOOL_EXECUTION_DISPATCH_METHODS = frozenset(
    {"execute", "execute_prepared", "execute_authorized"}
)
_TOOL_RUNTIME_STATE_MARKERS = ("permission", "authorization")


@dataclass(frozen=True, slots=True)
class _StaticImport:
    source_module: str
    symbol: str | None
    form: str
    line: int


def test_retired_prompt_and_session_assembly_modules_are_absent() -> None:
    assert not (PACKAGE_ROOT / "agent" / "prompts.py").exists()
    assert not (PACKAGE_ROOT / "session" / "projection.py").exists()
    assert not (PACKAGE_ROOT / "agent" / "session" / "projection.py").exists()
    agent_prompt_module = ".".join(("myclaw", "agent", "prompts"))
    assert importlib.util.find_spec(agent_prompt_module) is None
    for session_projection_module in (
        ".".join(("myclaw", "session", "projection")),
        ".".join(("myclaw", "agent", "session", "projection")),
    ):
        try:
            spec = importlib.util.find_spec(session_projection_module)
        except ModuleNotFoundError:
            spec = None
        assert spec is None


def test_feature_packages_have_canonical_locations() -> None:
    retired_modules = ("tools",)
    assert all(not (PACKAGE_ROOT / module).exists() for module in retired_modules)
    assert (PACKAGE_ROOT / "session" / "session.py").is_file()
    assert not tuple((PACKAGE_ROOT / "agent" / "session").glob("*.py"))
    for module in ("manager", "records", "store", "dream"):
        assert (PACKAGE_ROOT / "memory" / f"{module}.py").is_file()
        assert not (PACKAGE_ROOT / "agent" / "memory" / f"{module}.py").exists()

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util\n"
                "modules = ('myclaw.tools',)\n"
                "assert all(importlib.util.find_spec(module) is None for module in modules)\n"
                "assert importlib.util.find_spec('myclaw.session.session') is not None\n"
                "assert importlib.util.find_spec('myclaw.agent.session.session') is None\n"
                "assert importlib.util.find_spec('myclaw.memory.manager') is not None\n"
                "assert importlib.util.find_spec('myclaw.agent.memory.manager') is None\n"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def _imports(path: Path) -> tuple[tuple[str, int], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append((node.module, node.lineno))
    return tuple(imports)


def _resolved_from_module(node: ast.ImportFrom, *, package: tuple[str, ...]) -> str:
    if node.level:
        retained = len(package) - node.level + 1
        base = package[: max(0, retained)]
    else:
        base = ()
    if node.module is not None:
        base = (*base, *node.module.split("."))
    return ".".join(base)


def _resolved_imports(
    source: str,
    *,
    package: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    tree = ast.parse(source)
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        module = _resolved_from_module(node, package=package)
        if module:
            imports.append((module, node.lineno))
        imports.extend(
            (".".join((module, alias.name)) if module else alias.name, node.lineno)
            for alias in node.names
            if module or alias.name
        )
    return tuple(imports)


def _resolved_static_imports(
    source: str,
    *,
    package: tuple[str, ...],
) -> tuple[_StaticImport, ...]:
    tree = ast.parse(source)
    imports: list[_StaticImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                _StaticImport(alias.name, None, "import", node.lineno) for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            module = _resolved_from_module(node, package=package)
            imports.extend(
                _StaticImport(module, alias.name, "from", node.lineno) for alias in node.names
            )
    return tuple(imports)


def _imported_module_names(reference: _StaticImport) -> tuple[str, ...]:
    modules = [reference.source_module]
    if reference.form == "from" and reference.symbol is not None:
        modules.append(
            ".".join((reference.source_module, reference.symbol))
            if reference.source_module
            else reference.symbol
        )
    return tuple(modules)


def _is_tools_dependency(reference: _StaticImport) -> bool:
    return any(
        module == "myclaw.agent.tools" or module.startswith("myclaw.agent.tools.")
        for module in _imported_module_names(reference)
    )


def _terminal_tool_import_violations(
    sources: Mapping[Path, str],
    *,
    allowed_symbols: Mapping[Path, frozenset[tuple[str, str]]],
) -> tuple[str, ...]:
    violations: list[str] = []
    for path in sorted(sources, key=str):
        allowed_for_path = allowed_symbols.get(path, frozenset())
        for reference in _resolved_static_imports(
            sources[path],
            package=tuple(path.parent.parts),
        ):
            if not _is_tools_dependency(reference):
                continue
            is_allowed = (
                reference.form == "from"
                and reference.symbol is not None
                and (reference.source_module, reference.symbol) in allowed_for_path
            )
            if is_allowed:
                continue
            imported = reference.source_module
            if reference.form == "from" and reference.symbol is not None:
                imported = f"{imported}.{reference.symbol}" if imported else reference.symbol
            violations.append(f"{path}:{reference.line} imports {imported}")
    return tuple(violations)


def _tool_execution_dispatch_violations(
    sources: Mapping[Path, str],
    *,
    allowed_dispatchers: frozenset[Path],
) -> tuple[str, ...]:
    violations: list[str] = []
    for path in sorted(sources, key=str):
        if path in allowed_dispatchers:
            continue
        tree = ast.parse(sources[path], filename=str(path))
        typed_tool_names = _typed_tool_names(tree)
        display_path = path.relative_to(PROJECT_ROOT) if path.is_absolute() else path
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            if method not in _TOOL_EXECUTION_DISPATCH_METHODS:
                continue
            if method == "execute" and not _looks_like_tool_receiver(
                node.func.value,
                typed_tool_names=typed_tool_names,
            ):
                continue
            violations.append(f"{display_path}:{node.lineno} calls {method}")
    return tuple(violations)


def _typed_tool_names(tree: ast.AST) -> frozenset[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and _annotation_names_tool(node.annotation):
            names.add(node.arg)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and _annotation_names_tool(node.annotation)
        ):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign) and _constructs_tool(node.value):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return frozenset(names)


def _annotation_names_tool(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    return any(
        (isinstance(node, ast.Name) and (node.id == "BaseTool" or node.id.endswith("Tool")))
        or (
            isinstance(node, ast.Attribute)
            and (node.attr == "BaseTool" or node.attr.endswith("Tool"))
        )
        for node in ast.walk(annotation)
    )


def _constructs_tool(value: ast.expr) -> bool:
    if not isinstance(value, ast.Call):
        return False
    constructor = value.func
    return (
        isinstance(constructor, ast.Name)
        and (constructor.id == "BaseTool" or constructor.id.endswith("Tool"))
    ) or (
        isinstance(constructor, ast.Attribute)
        and (constructor.attr == "BaseTool" or constructor.attr.endswith("Tool"))
    )


def _looks_like_tool_receiver(
    receiver: ast.expr,
    *,
    typed_tool_names: frozenset[str],
) -> bool:
    if isinstance(receiver, ast.Name):
        return (
            receiver.id in typed_tool_names
            or receiver.id == "tool"
            or receiver.id.endswith("_tool")
        )
    if isinstance(receiver, ast.Attribute):
        return receiver.attr == "tool" or receiver.attr.endswith("_tool")
    if isinstance(receiver, ast.Call):
        return _constructs_tool(receiver)
    return (
        isinstance(receiver, ast.Subscript)
        and isinstance(receiver.value, ast.Attribute)
        and receiver.value.attr in {"catalog", "_tools"}
    )


def _base_tool_runtime_state_violations(sources: Mapping[Path, str]) -> tuple[str, ...]:
    violations: list[str] = []
    for path in sorted(sources, key=str):
        tree = ast.parse(sources[path], filename=str(path))
        display_path = path.relative_to(PROJECT_ROOT) if path.is_absolute() else path
        for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            if not any(
                (isinstance(base, ast.Name) and base.id == "BaseTool")
                or (isinstance(base, ast.Attribute) and base.attr == "BaseTool")
                for base in class_node.bases
            ):
                continue
            for node in ast.walk(class_node):
                if not (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    continue
                if any(marker in node.attr.casefold() for marker in _TOOL_RUNTIME_STATE_MARKERS):
                    violations.append(
                        f"{display_path}:{node.lineno} stores {node.attr} on {class_node.name}"
                    )
    return tuple(violations)


def _base_tool_prepare_contract_violations(sources: Mapping[Path, str]) -> tuple[str, ...]:
    violations: list[str] = []
    for path in sorted(sources, key=str):
        tree = ast.parse(sources[path], filename=str(path))
        display_path = path.relative_to(PROJECT_ROOT) if path.is_absolute() else path
        for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            prepare = next(
                (
                    node
                    for node in class_node.body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "prepare"
                ),
                None,
            )
            if class_node.name == "BaseTool":
                if prepare is None:
                    violations.append(f"{display_path}:{class_node.lineno} lacks BaseTool.prepare")
                    continue
                decorators = {
                    decorator.id
                    for decorator in prepare.decorator_list
                    if isinstance(decorator, ast.Name)
                }
                returns = None if prepare.returns is None else ast.unparse(prepare.returns)
                if "final" not in decorators or returns != "ToolInvocationFacts":
                    violations.append(
                        f"{display_path}:{prepare.lineno} has an invalid BaseTool.prepare contract"
                    )
                continue
            is_base_tool_subclass = any(
                (isinstance(base, ast.Name) and base.id == "BaseTool")
                or (isinstance(base, ast.Attribute) and base.attr == "BaseTool")
                for base in class_node.bases
            )
            if is_base_tool_subclass and prepare is not None:
                violations.append(
                    f"{display_path}:{prepare.lineno} overrides final BaseTool.prepare"
                )
    return tuple(violations)


def _retired_mcp_runtime_import_violations(sources: Mapping[Path, str]) -> tuple[str, ...]:
    violations: list[str] = []
    for path in sorted(sources, key=str):
        for reference in _resolved_static_imports(
            sources[path],
            package=tuple(path.parent.parts),
        ):
            imported_modules = _imported_module_names(reference)
            if not any(
                module == "myclaw.mcp_runtime" or module.startswith("myclaw.mcp_runtime.")
                for module in imported_modules
            ):
                continue
            violations.append(f"{path}:{reference.line} imports {imported_modules[-1]}")
    return tuple(violations)


def _is_blackboard_module(module: str) -> bool:
    return module == "myclaw.agent.blackboard" or module.startswith("myclaw.agent.blackboard.")


@pytest.mark.parametrize(
    "source",
    [
        "import myclaw.agent.blackboard",
        "from myclaw.agent.blackboard import Blackboard",
        "from myclaw.agent import blackboard",
        "from .blackboard import Blackboard",
        "from . import blackboard",
        "def load():\n    import myclaw.agent.blackboard",
        "if TYPE_CHECKING:\n    from . import blackboard",
    ],
)
def test_import_scanner_resolves_blackboard_dependency_forms(source: str) -> None:
    assert any(
        _is_blackboard_module(module)
        for module, _ in _resolved_imports(source, package=("myclaw", "agent"))
    )


def test_production_code_does_not_import_removed_contracts_package() -> None:
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for path in _python_files(PACKAGE_ROOT)
        for module, line in _imports(path)
        if module == "myclaw.contracts" or module.startswith("myclaw.contracts.")
    ]

    assert violations == []


def test_production_tool_execution_dispatch_has_one_gateway_boundary() -> None:
    sources = {
        path: path.read_text(encoding="utf-8") for path in _python_files(PACKAGE_ROOT)
    }
    violations = _tool_execution_dispatch_violations(
        sources,
        allowed_dispatchers=frozenset(
            {
                PACKAGE_ROOT / "agent" / "tools" / "base.py",
                PACKAGE_ROOT / "agent" / "tools" / "tool_gateway.py",
            }
        ),
    )

    assert violations == ()


def test_production_tools_have_no_legacy_string_authorization_surface() -> None:
    """Keep authorization decisions on typed invocation facts and one policy."""
    forbidden_identifiers = {
        "check_safety",
        "workspace_path_safety_reason",
        "safety_reason",
        "safety_assessment",
        "legacy_safety_reason",
        "_LegacyAuthorizationSession",
        "requires_confirmation",
        "requires_legacy_destructive_confirmation",
        "authorization_callback",
        "_authorization_callback",
    }
    violations: list[str] = []
    for path in _python_files(PACKAGE_ROOT / "agent" / "tools"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            identifier: str | None = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                identifier = node.name
            elif isinstance(node, ast.Name):
                identifier = node.id
            elif isinstance(node, ast.arg):
                identifier = node.arg
            elif isinstance(node, ast.keyword):
                identifier = node.arg
            elif isinstance(node, ast.Attribute):
                identifier = node.attr
            if identifier in forbidden_identifiers:
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{getattr(node, 'lineno', 0)}:{identifier}"
                )

    assert violations == []


def test_base_tool_prepare_is_the_final_structured_fact_seam() -> None:
    sources = {
        path: path.read_text(encoding="utf-8") for path in _python_files(PACKAGE_ROOT)
    }

    assert _base_tool_prepare_contract_violations(sources) == ()


def test_shared_tools_store_no_permission_or_authorization_runtime_state() -> None:
    sources = {
        path: path.read_text(encoding="utf-8") for path in _python_files(PACKAGE_ROOT)
    }

    assert _base_tool_runtime_state_violations(sources) == ()


@pytest.mark.parametrize("method", sorted(_TOOL_EXECUTION_DISPATCH_METHODS))
def test_tool_execution_dispatch_checker_rejects_gateway_bypasses(method: str) -> None:
    path = Path("myclaw/agent/bypass.py")
    source = f"async def bypass(tool):\n    await tool.{method}({{}})"

    assert _tool_execution_dispatch_violations(
        {path: source},
        allowed_dispatchers=frozenset(),
    ) == (f"{path}:2 calls {method}",)


def test_tool_execution_dispatch_checker_allows_unrelated_execute_methods() -> None:
    path = Path("myclaw/agent/host_adapter.py")
    source = "async def run(adapter):\n    await adapter.execute()"

    assert (
        _tool_execution_dispatch_violations(
            {path: source},
            allowed_dispatchers=frozenset(),
        )
        == ()
    )


def test_tool_execution_dispatch_checker_uses_tool_types_not_only_names() -> None:
    path = Path("myclaw/agent/bypass.py")
    source = "async def bypass(candidate: BaseTool):\n    await candidate.execute()"

    assert _tool_execution_dispatch_violations(
        {path: source},
        allowed_dispatchers=frozenset(),
    ) == (f"{path}:2 calls execute",)


@pytest.mark.parametrize("attribute", ["_permission_context", "_authorization_session"])
def test_shared_tool_state_checker_rejects_permission_runtime_state(attribute: str) -> None:
    path = Path("myclaw/agent/unsafe_tool.py")
    source = (
        "class UnsafeTool(BaseTool):\n"
        "    def __init__(self, value):\n"
        f"        self.{attribute} = value\n"
    )

    assert _base_tool_runtime_state_violations({path: source}) == (
        f"{path}:3 stores {attribute} on UnsafeTool",
    )


def test_prepare_contract_checker_rejects_tuple_and_subclass_override() -> None:
    path = Path("myclaw/agent/unsafe_tool.py")
    source = (
        "class BaseTool:\n"
        "    async def prepare(self, arguments) -> tuple[dict, str | None]: ...\n"
        "class UnsafeTool(BaseTool):\n"
        "    async def prepare(self, arguments) -> ToolInvocationFacts: ...\n"
    )

    assert len(_base_tool_prepare_contract_violations({path: source})) == 2


@pytest.mark.parametrize("root", [PACKAGE_ROOT / "utils", PACKAGE_ROOT / "errors.py"])
def test_foundation_modules_do_not_import_domain_modules(root: Path) -> None:
    files = (root,) if root.is_file() else _python_files(root)
    allowed = {"myclaw.errors", "myclaw.utils"}
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for path in files
        for module, line in _imports(path)
        if module.startswith("myclaw.")
        and not any(module == prefix or module.startswith(f"{prefix}.") for prefix in allowed)
    ]

    assert violations == []


def test_tools_do_not_depend_on_provider() -> None:
    forbidden = {"myclaw.provider"}
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for path in _python_files(PACKAGE_ROOT / "agent" / "tools")
        for module, line in _imports(path)
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden)
    ]

    assert violations == []


def test_terminal_depends_on_ports_instead_of_tool_implementations() -> None:
    sources = {
        path.relative_to(PROJECT_ROOT): path.read_text(encoding="utf-8")
        for path in _python_files(PACKAGE_ROOT / "terminal")
    }

    violations = _terminal_tool_import_violations(
        sources,
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations == ()


@pytest.mark.parametrize(("module", "symbol"), sorted(_CLI_TOOL_IMPORTS))
def test_terminal_tool_import_checker_allows_each_cli_symbol(
    module: str,
    symbol: str,
) -> None:
    violations = _terminal_tool_import_violations(
        {_CLI_PATH: f"from {module} import {symbol}"},
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations == ()


@pytest.mark.parametrize(
    "source",
    [
        "from myclaw.agent.tools.mcp_runtime import MCPRuntimeManager as Manager",
        "from ..agent.tools.mcp_runtime import MCPRuntimeManager",
        "from ..agent.tools.tool_gateway import BUILT_IN_TOOL_NAMES as BUILT_INS",
    ],
)
def test_terminal_tool_import_checker_resolves_allowed_aliases_and_relative_imports(
    source: str,
) -> None:
    violations = _terminal_tool_import_violations(
        {_CLI_PATH: source},
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations == ()


def test_terminal_tool_import_checker_retains_original_symbol_form_and_line() -> None:
    references = _resolved_static_imports(
        "\nfrom myclaw.agent.tools.mcp_runtime import MCPRuntimeManager as Manager",
        package=("myclaw", "terminal"),
    )

    assert references == (
        _StaticImport("myclaw.agent.tools.mcp_runtime", "MCPRuntimeManager", "from", 2),
    )


@pytest.mark.parametrize(
    "path",
    [
        Path("myclaw/terminal/conversation.py"),
        Path("myclaw/terminal/process_entry.py"),
        Path("myclaw/terminal/internal/loader.py"),
    ],
)
def test_terminal_tool_import_checker_rejects_cli_symbols_outside_cli(path: Path) -> None:
    violations = _terminal_tool_import_violations(
        {path: "from myclaw.agent.tools.mcp_runtime import MCPRuntimeManager"},
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations


@pytest.mark.parametrize(
    "source",
    [
        "from myclaw.agent.tools.tool_gateway import ToolGateway",
        "from myclaw.agent.tools.tool_gateway import BUILT_IN_TOOL_NAMES, ToolGateway",
        "from myclaw.agent.tools.mcp_runtime import allocate_mcp_tool_name",
        "import myclaw.agent.tools.mcp_runtime",
        "import myclaw.agent.tools.mcp_runtime as runtime",
        "from myclaw.agent.tools import mcp_runtime",
        "from myclaw.agent.tools.mcp_runtime import *",
        "from myclaw.agent.tools.tool_gateway import ToolGateway as Gateway",
        "from ..agent.tools.tool_gateway import ToolGateway",
        "def load():\n    from myclaw.agent.tools.mcp_runtime import allocate_mcp_tool_name",
        "if TYPE_CHECKING:\n    from myclaw.agent.tools.mcp_runtime import allocate_mcp_tool_name",
    ],
)
def test_terminal_tool_import_checker_rejects_unapproved_tool_imports(source: str) -> None:
    violations = _terminal_tool_import_violations(
        {_CLI_PATH: source},
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations


@pytest.mark.parametrize(
    "source",
    [
        "import asyncio",
        "from myclaw.management.service import ManagementViewService",
        "from .conversation import TerminalConversationApp",
    ],
)
def test_terminal_tool_import_checker_allows_non_tool_dependencies(source: str) -> None:
    violations = _terminal_tool_import_violations(
        {_CLI_PATH: source},
        allowed_symbols={_CLI_PATH: _CLI_TOOL_IMPORTS},
    )

    assert violations == ()


def test_retired_mcp_runtime_export_is_absent() -> None:
    assert not (PACKAGE_ROOT / "mcp_runtime.py").exists()

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; print(importlib.util.find_spec('myclaw.mcp_runtime'))",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert probe.stdout.strip() == "None"


@pytest.mark.parametrize(
    "source",
    [
        "import myclaw.mcp_runtime",
        "from myclaw.mcp_runtime import MCPRuntimeManager",
        "from myclaw import mcp_runtime",
        "from ..mcp_runtime import MCPRuntimeManager",
        "from .. import mcp_runtime",
    ],
)
def test_retired_mcp_runtime_import_checker_covers_import_forms(source: str) -> None:
    violations = _retired_mcp_runtime_import_violations({_CLI_PATH: source})

    assert violations


def test_production_code_does_not_import_retired_mcp_runtime_export() -> None:
    sources = {
        path.relative_to(PROJECT_ROOT): path.read_text(encoding="utf-8")
        for path in _python_files(PACKAGE_ROOT)
    }

    violations = _retired_mcp_runtime_import_violations(sources)

    assert violations == ()


def test_context_builder_constructor_owns_only_context_dependencies() -> None:
    parameters = inspect.signature(ContextBuilder.__init__).parameters

    assert tuple(parameters) == (
        "self",
        "workspace",
        "timezone_name",
        "agent_home",
        "memory_manager",
        "skill_loader",
    )
    assert parameters["agent_home"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["memory_manager"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["skill_loader"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "clock" not in parameters
    assert "tool_gateway" not in parameters
    assert "tool_schema" not in parameters
    assert "router" not in parameters
    assert "model_router" not in parameters


def test_mcp_keyword_module_does_not_import_private_configuration_implementation() -> None:
    path = PACKAGE_ROOT / "agent" / "tools" / "mcp_keywords.py"
    source = path.read_text(encoding="utf-8")
    private_configuration_imports = [
        reference
        for reference in _resolved_static_imports(
            source,
            package=("myclaw", "agent", "tools"),
        )
        if reference.source_module == "myclaw.config.config"
        and reference.symbol is not None
        and reference.symbol.startswith("_")
    ]

    assert private_configuration_imports == []


def test_context_builder_does_not_import_model_request_runtime_boundaries() -> None:
    path = PACKAGE_ROOT / "agent" / "context.py"
    forbidden_prefixes = (
        "myclaw.provider",
        "myclaw.router",
        "myclaw.agent.tools",
    )
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for module, line in _imports(path)
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden_prefixes)
    ]

    assert violations == []


def test_agent_loop_control_interface_is_consolidated() -> None:
    path = PACKAGE_ROOT / "agent" / "loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    assert [node for node in classes if node.name == "AgentLoopControl"] == []

    terminal_controls = [node for node in classes if node.name == "TerminalAgentLoopControl"]
    assert len(terminal_controls) == 1
    terminal_control = terminal_controls[0]
    assert len(terminal_control.bases) == 1
    assert isinstance(terminal_control.bases[0], ast.Name)
    assert terminal_control.bases[0].id == "Protocol"
    assert {
        node.name
        for node in terminal_control.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    } == {
        "has_active_run",
        "foreground_input_admitted",
        "cancel_active_run",
        "bind_confirmation_callback",
        "respond_to_confirmation",
        "project_foreground_conversation",
    }

    exports = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    assert isinstance(exports, (ast.List, ast.Tuple))
    assert "AgentLoopControl" not in {
        element.value
        for element in exports.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }


def test_agent_loop_does_not_retain_a_title_prompt_outside_context_builder() -> None:
    path = PACKAGE_ROOT / "agent" / "loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    retained_title_prompts = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "_title_prompt"
    ]

    assert retained_title_prompts == []


def test_agent_loop_delegates_foreground_context_construction_to_context_builder() -> None:
    path = PACKAGE_ROOT / "agent" / "loop.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    agent_loop = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentLoop"
    )
    methods = {
        node.name: node
        for node in agent_loop.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }

    def calls_builder(method_name: str, builder_method: str) -> bool:
        method = methods[method_name]
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == builder_method
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_context_builder"
            for node in ast.walk(method)
        )

    assert "render_template" not in source
    assert "build_messages" not in source
    assert "_project_foreground_messages" not in source
    assert "_project_foreground_summary_messages" not in methods
    assert calls_builder("_execute_foreground_logged", "build_foreground_messages")


def test_agent_loop_delegates_schedule_context_construction_to_context_builder() -> None:
    path = PACKAGE_ROOT / "agent" / "loop.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    agent_loop = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentLoop"
    )
    prepare_schedule = next(
        node
        for node in agent_loop.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_schedule_agent_scoped"
    )

    assert hasattr(ContextBuilder, "build_schedule_messages")
    assert "_project_schedule_messages" not in source
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_schedule_messages"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_context_builder"
        for node in ast.walk(prepare_schedule)
    )


def test_agent_loop_request_paths_stay_inside_context_builder() -> None:
    path = PACKAGE_ROOT / "agent" / "loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    agent_loop = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentLoop"
    )
    methods = {
        node.name: node
        for node in agent_loop.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }
    expected_builder_calls = {
        "_execute_foreground_logged": "build_foreground_messages",
        "_run_schedule_agent_scoped": "build_schedule_messages",
        "_router_stream_title": "build_title_messages",
    }
    for method_name, builder_method in expected_builder_calls.items():
        method = methods[method_name]
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == builder_method
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_context_builder"
            for node in ast.walk(method)
        )

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_validate_model_context_budget"
        for node in ast.walk(methods["preflight"])
    )


def test_issue_243_production_model_calls_use_one_explicit_run_context_seam() -> None:
    path = PACKAGE_ROOT / "agent" / "loop.py"
    loop_tree = ast.parse(path.read_text(encoding="utf-8"))
    tree = loop_tree
    loop_class = next(
        node
        for node in loop_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AgentLoop"
    )
    methods = {
        node.name: node
        for node in loop_class.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }
    functions = {
        node.name: node
        for node in loop_tree.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }

    for method_name in ("_execute_foreground_logged", "_run_schedule_agent_scoped"):
        runner_call = next(
            node
            for node in ast.walk(methods[method_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "runner"
        )
        assert not {"model_router", "request_preparer"} & {
            keyword.arg for keyword in runner_call.keywords
        }

    compactor_tree = ast.parse(
        (PACKAGE_ROOT / "agent" / "memory" / "conversation_compactor.py").read_text(
            encoding="utf-8"
        )
    )
    assert not any(
        isinstance(node, ast.ClassDef) and node.name == "ConversationCompactor"
        for node in compactor_tree.body
    )
    adapter = next(
        node
        for node in compactor_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AgentRunContextRouterAdapter"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"getattr", "hasattr"}
        for node in ast.walk(adapter)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_foreground_runtime_status_input"
        for node in ast.walk(methods["_validate_model_context_budget"])
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_foreground_runtime_status_input"
        for node in ast.walk(methods["runtime_status_input"])
    )
    status_helper = functions["_foreground_runtime_status_input"]
    status_builder_calls = {
        node.func.attr
        for node in ast.walk(status_helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "context_builder"
    }
    assert "build_status_messages" in status_builder_calls
    removed_builder_methods = {
        "_build_status_messages_for_skills",
        "_foreground_system_prompt_for_skills",
    }
    assert all(not hasattr(ContextBuilder, name) for name in removed_builder_methods)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in removed_builder_methods
        for node in ast.walk(tree)
    )
    assert "skill_state" not in {argument.arg for argument in status_helper.args.kwonlyargs}
    assert "render_template" not in path.read_text(encoding="utf-8")


def test_runner_summary_and_dream_keep_context_builder_out_of_their_boundaries() -> None:
    paths = (
        PACKAGE_ROOT / "agent" / "runner.py",
        PACKAGE_ROOT / "agent" / "memory" / "conversation_compactor.py",
        PACKAGE_ROOT / "memory" / "dream.py",
    )
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for path in paths
        for module, line in _imports(path)
        if module == "myclaw.agent.context" or module.startswith("myclaw.agent.context.")
    ]
    assert violations == []


def test_agent_modules_do_not_depend_on_terminal_presentation() -> None:
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line} imports {module}"
        for path in _python_files(PACKAGE_ROOT / "agent")
        for module, line in _imports(path)
        if module == "myclaw.terminal" or module.startswith("myclaw.terminal.")
    ]

    assert violations == []


def test_cli_exclusively_owns_the_runtime_confirmation_coordinator() -> None:
    constructor_sites: list[Path] = []
    for path in _python_files(PACKAGE_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "ToolConfirmationCoordinator")
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "ToolConfirmationCoordinator"
                )
            )
            for node in ast.walk(tree)
        ):
            constructor_sites.append(path.relative_to(PROJECT_ROOT))

    assert constructor_sites == [_CLI_PATH]

    loop_path = PACKAGE_ROOT / "agent" / "loop.py"
    terminal_path = PACKAGE_ROOT / "terminal" / "conversation.py"
    assert "ToolConfirmationCoordinator" not in loop_path.read_text(encoding="utf-8")
    assert "ToolConfirmationCoordinator" not in terminal_path.read_text(encoding="utf-8")

    terminal_tree = ast.parse(terminal_path.read_text(encoding="utf-8"), filename=str(terminal_path))
    terminal_app = next(
        node
        for node in terminal_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TerminalConversationApp"
    )
    bind = next(
        node
        for node in terminal_app.body
        if isinstance(node, ast.FunctionDef) and node.name == "bind_confirmation_coordinator"
    )
    coordinator_parameter = bind.args.args[1]
    assert isinstance(coordinator_parameter.annotation, ast.Name)
    assert coordinator_parameter.annotation.id == "ConfirmationPresentationCoordinator"

    forbidden_queue_names = {
        "_pending_confirmation",
        "_confirmation_queue",
        "_foreground_confirmations",
        "_background_confirmations",
    }
    for path in (loop_path, terminal_path):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        retained = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in forbidden_queue_names
        }
        assert retained == set()


def test_terminal_conversation_lifecycle_has_no_business_lifecycle_calls() -> None:
    path = PACKAGE_ROOT / "terminal" / "conversation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    app = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TerminalConversationApp"
    )
    lifecycle_methods = {"__init__", "on_mount", "on_unmount", "rebind_agent_loop"}
    business_calls = {
        "abort",
        "abort_and_wait",
        "close",
        "drain_inbound",
        "drain_outbound",
        "reset",
        "start",
    }
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{call.lineno} calls {call.func.attr}"
        for method in app.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and method.name in lifecycle_methods
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in business_calls
    ]

    assert violations == []


def test_package_initializers_do_not_create_aggregate_import_entries() -> None:
    violations: list[str] = []
    for path in _python_files(PACKAGE_ROOT):
        if path.name != "__init__.py" or path.parent == PACKAGE_ROOT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            else:
                continue
            violations.extend(
                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} imports {module}"
                for module in modules
                if module == "myclaw" or module.startswith("myclaw.")
            )

    assert violations == []


def test_host_selection_is_confined_to_the_workspace_filesystem_adapter() -> None:
    expected = {
        Path("myclaw/utils/host_filesystem.py"),
        Path("myclaw/agent/tools/core/exec_host.py"),
    }
    actual = {
        path.relative_to(PROJECT_ROOT)
        for path in _python_files(PACKAGE_ROOT)
        if any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr == "name"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        )
    }

    assert actual == expected


def test_superseded_tool_modules_are_absent() -> None:
    removed = (
        Path("myclaw/agent/tools/files/__init__.py"),
        Path("myclaw/agent/tools/files/file_tools.py"),
        Path("myclaw/agent/tools/security.py"),
        Path("myclaw/agent/tools/shell/__init__.py"),
        Path("myclaw/agent/tools/shell/owned_process.py"),
        Path("myclaw/agent/tools/shell/shell_tool.py"),
        Path("myclaw/agent/tools/web/__init__.py"),
        Path("myclaw/agent/tools/web/web_fetch.py"),
        Path("myclaw/agent/tools/web/web_search.py"),
        Path("myclaw/agent/tools/tool_artifacts.py"),
    )

    assert all(not (PROJECT_ROOT / path).exists() for path in removed)
