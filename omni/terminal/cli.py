"""Command-line entry point for Omni."""

import asyncio
import webbrowser
from pathlib import Path
from typing import cast

import typer
from rich.console import Console

from omni.agent.loop import ModelContextOverflowError
from omni.agent.workspace_state import (
    WorkspaceStateError,
)
from omni.config.agent_home import AgentHome
from omni.config.config import ConfigError, ConfigLoader
from omni.errors import MODEL_CONTEXT_OVERFLOW_MESSAGE, ErrorInfo
from omni.management.commands import ManagementCommandDispatcher
from omni.management.service import (
    FatalManagementError,
)
from omni.service.client import ServiceClient, ServiceStartupError
from omni.service.errors import ServiceError
from omni.terminal.conversation import (
    TerminalConversationApp,
    is_interactive_terminal,
)

app = typer.Typer(
    add_completion=False,
    help="Omni Personal Agent runtime.",
    rich_markup_mode="rich",
)
console = Console()

service_app = typer.Typer(add_completion=False, help="Manage the local Omni service.")
app.add_typer(service_app, name="service")

_MODEL_CONTEXT_OVERFLOW_ERROR = ErrorInfo(
    "model_context_overflow",
    MODEL_CONTEXT_OVERFLOW_MESSAGE,
)
_WORKSPACE_STATE_INITIALIZATION_ERROR = ErrorInfo(
    "persistence_error",
    "Workspace State could not be initialized at the reserved path.",
)
_TARGET_SESSION_PREPARATION_ERROR = ErrorInfo(
    "persistence_error",
    "Conversation Session could not be prepared.",
)
_RUNTIME_SESSION_REPLACEMENT_ERROR = ErrorInfo(
    "persistence_error",
    "Runtime Session replacement could not be completed.",
)
_RUNTIME_STARTUP_ERROR = ErrorInfo(
    "persistence_error",
    "Omni runtime could not be started.",
)
_RESTORE_STARTUP_ERROR = ErrorInfo(
    "persistence_error",
    "Workspace Restore could not be recovered.",
)
_SAFE_FATAL_MANAGEMENT_ERRORS = (
    _MODEL_CONTEXT_OVERFLOW_ERROR,
    _TARGET_SESSION_PREPARATION_ERROR,
    _RUNTIME_SESSION_REPLACEMENT_ERROR,
    _RESTORE_STARTUP_ERROR,
)


def _print_error_info(error: ErrorInfo | ServiceStartupError | ServiceError) -> None:
    console.print(
        f"{error.code}: {error.message}",
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


def _print_error(error: ErrorInfo, path: object) -> None:
    _print_error_info(error)
    console.print(f"Path: {path}", markup=False, highlight=False, soft_wrap=True)


def _approved_error_info(
    error: Exception,
    *,
    approved: tuple[ErrorInfo, ...],
    fallback: ErrorInfo,
) -> ErrorInfo:
    """Return only an exact, approved safe value from a domain exception."""
    candidate = vars(error).get("error")
    if type(candidate) is ErrorInfo and candidate in approved:
        return candidate
    return fallback


async def _run_service_cli_conversation(
    *,
    agent_home: AgentHome,
    workspace: Path,
) -> None:
    """Run the terminal as a client of the shared local service."""
    client = await ServiceClient.connect_or_start(agent_home, workspace)
    try:
        terminal_app = TerminalConversationApp(
            bus=client.bus,
            control=client.control,
            management_dispatcher=cast(ManagementCommandDispatcher, client.management_dispatcher),
        )
        terminal_app.bind_confirmation_coordinator(client.confirmation)
        await terminal_app.run_async()
        fatal_management_error = getattr(terminal_app, "fatal_management_error", None)
        if isinstance(fatal_management_error, FatalManagementError):
            raise fatal_management_error
    finally:
        await client.close()


async def _create_web_launch_url() -> str:
    client = await ServiceClient.connect_or_start(
        AgentHome.production(),
        Path.cwd(),
        attach_workspace=False,
    )
    try:
        return await client.create_web_ticket()
    finally:
        await client.close()


@app.callback(invoke_without_command=True)
def main(context: typer.Context) -> None:
    """Start the Omni Personal Agent."""
    if context.invoked_subcommand is not None:
        return
    agent_home = AgentHome.production()
    loader = ConfigLoader(agent_home)
    try:
        loader.load_for_startup()
    except ConfigError as config_error:
        _print_error(config_error.error, loader.path)
        exit_code = 1 if config_error.error.code == "persistence_error" else 2
        raise typer.Exit(code=exit_code) from None
    except OSError:
        _print_error(
            ErrorInfo("persistence_error", "User Configuration could not be read or written."),
            loader.path,
        )
        raise typer.Exit(code=1) from None
    if not is_interactive_terminal():
        _print_error_info(
            ErrorInfo(
                "interactive_terminal_required",
                "Terminal Conversation requires interactive stdin, stdout, and stderr TTYs.",
            )
        )
        raise typer.Exit(code=2)
    if loader.diagnostics:
        console.print(
            "".join(f"{diagnostic.message}\n" for diagnostic in loader.diagnostics),
            markup=False,
            highlight=False,
            soft_wrap=True,
            end="",
        )
    try:
        asyncio.run(
            _run_service_cli_conversation(
                agent_home=loader.agent_home,
                workspace=Path.cwd(),
            )
        )
    except WorkspaceStateError:
        _print_error_info(_WORKSPACE_STATE_INITIALIZATION_ERROR)
        raise typer.Exit(code=1) from None
    except ModelContextOverflowError as context_error:
        _print_error_info(
            _approved_error_info(
                context_error,
                approved=(_MODEL_CONTEXT_OVERFLOW_ERROR,),
                fallback=_RUNTIME_STARTUP_ERROR,
            )
        )
        raise typer.Exit(code=1) from None
    except FatalManagementError as fatal_error:
        _print_error_info(
            _approved_error_info(
                fatal_error,
                approved=_SAFE_FATAL_MANAGEMENT_ERRORS,
                fallback=_RUNTIME_STARTUP_ERROR,
            )
        )
        raise typer.Exit(code=1) from None
    except ServiceStartupError as service_error:
        _print_error_info(service_error)
        raise typer.Exit(code=1) from None
    except ServiceError as service_error:
        _print_error_info(service_error)
        raise typer.Exit(code=1) from None
    except Exception:
        _print_error_info(_RUNTIME_STARTUP_ERROR)
        raise typer.Exit(code=1) from None


@service_app.command("stop")
def service_stop_command() -> None:
    """Request graceful shutdown of the current local service."""
    try:
        stopped = asyncio.run(ServiceClient.stop_existing(AgentHome.production()))
    except ServiceStartupError as error:
        _print_error_info(error)
        raise typer.Exit(code=1) from None
    if not stopped:
        _print_error_info(
            ServiceStartupError("service_not_running", "No active local service was found.")
        )
        raise typer.Exit(code=1)


@app.command("web")
def web_command() -> None:
    """Open the authenticated local Web Interface."""
    try:
        url = asyncio.run(_create_web_launch_url())
        if not webbrowser.open_new_tab(url):
            console.print(url, markup=False, highlight=False)
    except ServiceStartupError as error:
        _print_error_info(error)
        raise typer.Exit(code=1) from None
    except ServiceError as error:
        _print_error_info(error)
        raise typer.Exit(code=1) from None


@app.command("config")
def config_command() -> None:
    """Display User Configuration with plaintext API keys redacted."""
    agent_home = AgentHome.production()
    loader = ConfigLoader(agent_home)
    try:
        loader.ensure_default()
        view = loader.view()
    except (OSError, UnicodeError):
        _print_error(
            ErrorInfo("persistence_error", "User Configuration could not be read or written."),
            loader.path,
        )
        raise typer.Exit(code=1) from None

    console.print(
        view.header_text(),
        markup=False,
        highlight=False,
        soft_wrap=True,
        end="",
    )
    console.print(
        view.redacted_content,
        markup=False,
        highlight=False,
        soft_wrap=True,
        end="" if view.redacted_content.endswith("\n") else "\n",
    )
    if view.error is not None:
        raise typer.Exit(code=2)
