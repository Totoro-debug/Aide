"""Compose per-Agent-Run deferred Tool exposure and local Tool Search."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from myclaw.agent.tools.base import BaseTool
from myclaw.agent.tools.core.tool_search import ToolSearchTool
from myclaw.agent.tools.search import (
    BUILTIN_TOOL_SEARCH_KEYWORDS,
    ToolSearchDocument,
    ToolSearchIndex,
)
from myclaw.agent.tools.tool_gateway import ToolGateway
from myclaw.permission.policy import PermissionContext, PermissionSnapshot

RUN_BASELINE_TOOL_NAMES: tuple[str, ...] = (
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob",
    "grep",
    "exec",
    "tool_search",
)


def build_agent_run_gateway(
    gateway: ToolGateway,
    *,
    excluded_names: Collection[str] = (),
    mcp_keywords: Mapping[str, Sequence[str]] | None = None,
    permission_snapshot: PermissionSnapshot | None = None,
    permission_context: PermissionContext | None = None,
) -> ToolGateway:
    """Build one isolated deferred-exposure Gateway view.

    The index is fixed from the resulting complete Catalog. Only the Gateway's
    exposure set changes when the injected Tool Search callback runs.
    """
    if not isinstance(gateway, ToolGateway):
        raise TypeError("Agent Run Tool Gateway requires a ToolGateway")
    keyword_mapping = {} if mcp_keywords is None else mcp_keywords
    if not isinstance(keyword_mapping, Mapping):
        raise TypeError("Agent Run MCP keywords must be a mapping")

    run_gateway: ToolGateway | None = None
    search_index: ToolSearchIndex | None = None

    def search_and_activate(query: str) -> tuple[str, ...]:
        if run_gateway is None or search_index is None:
            raise RuntimeError("Agent Run Tool Search is not initialized")
        names = search_index.search(query, excluded_names=run_gateway.exposed_names)
        run_gateway.expose(names)
        return names

    run_gateway = gateway.for_run(
        excluded_names=excluded_names,
        exposed_names=RUN_BASELINE_TOOL_NAMES,
        run_tools=(ToolSearchTool(search_and_activate),),
        permission_snapshot=permission_snapshot,
        permission_context=permission_context,
    )
    baseline_names = frozenset(RUN_BASELINE_TOOL_NAMES)
    documents = tuple(
        _document_for_tool(
            tool,
            catalog_order=catalog_order,
            mcp_keywords=keyword_mapping,
        )
        for catalog_order, tool in enumerate(run_gateway.catalog)
        if tool.name not in baseline_names
    )
    search_index = ToolSearchIndex(documents)
    return run_gateway


def _document_for_tool(
    tool: BaseTool,
    *,
    catalog_order: int,
    mcp_keywords: Mapping[str, Sequence[str]],
) -> ToolSearchDocument:
    original_name = getattr(tool, "remote_name", tool.name)
    if not isinstance(original_name, str) or not original_name:
        original_name = tool.name

    builtin_keywords = BUILTIN_TOOL_SEARCH_KEYWORDS.get(tool.name, ())
    configured_keywords = mcp_keywords.get(tool.name, ())
    if isinstance(configured_keywords, (str, bytes)):
        raise TypeError("Agent Run MCP keywords must contain string sequences")
    keyword_terms = tuple(configured_keywords)
    if keyword_terms == (original_name,):
        keyword_terms = ()
    terms = (original_name, *builtin_keywords, *keyword_terms)
    return ToolSearchDocument(
        name=tool.name,
        terms=terms,
        catalog_order=catalog_order,
    )


__all__ = ["RUN_BASELINE_TOOL_NAMES", "build_agent_run_gateway"]
