"""Independent Tool Search retrieval and BaseTool boundary tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from aide.agent.tools.base import ToolError
from aide.agent.tools.core.tool_search import ToolSearchTool
from aide.agent.tools.search import (
    BM25_B,
    BM25_K1,
    BUILTIN_TOOL_SEARCH_KEYWORDS,
    MAX_TOOL_SEARCH_RESULTS,
    ToolSearchDocument,
    ToolSearchIndex,
)


def _document(name: str, text: str, catalog_order: int) -> ToolSearchDocument:
    return ToolSearchDocument(
        name=name,
        terms=tuple(text.split()),
        catalog_order=catalog_order,
    )


def test_document_normalizes_terms_and_remains_immutable() -> None:
    document = ToolSearchDocument(
        name="model_name",
        terms=(" Original_Name ", "original-name", "Pull Request", "", "  "),
        catalog_order=4,
    )

    assert document.name == "model_name"
    assert document.terms == ("original_name", "original-name", "pull request")
    with pytest.raises(FrozenInstanceError):
        document.name = "changed"  # type: ignore[misc]


def test_empty_and_nonmatching_queries_return_no_names() -> None:
    index = ToolSearchIndex((_document("reader", "read file", 0),))

    assert index.search("") == ()
    assert index.search("!!!") == ()
    assert index.search("calendar") == ()


def test_empty_corpora_are_safe_and_repeated_construction_is_deterministic() -> None:
    assert ToolSearchIndex(()).search("anything") == ()
    assert ToolSearchIndex((_document("empty", "", 0),)).search("anything") == ()

    documents = (
        _document("later", "same", 9),
        _document("earlier", "same", 2),
    )
    results = tuple(ToolSearchIndex(documents).search("same") for _ in range(3))

    assert results == (("earlier", "later"),) * 3


def test_search_normalizes_queries_and_returns_the_exact_model_name() -> None:
    index = ToolSearchIndex((_document("mcp_github_search", "search issues", 0),))

    assert index.search("ISSUES-search") == ("mcp_github_search",)


def test_bm25_hand_calculated_match_is_finite_and_deterministic() -> None:
    index = ToolSearchIndex(
        (
            _document("tf_heavy", "alpha alpha common", 0),
            _document("length_heavy", "alpha common common common", 1),
        )
    )

    assert (BM25_K1, BM25_B) == (1.2, 0.75)
    # Hand calculation: both terms have df=2, and alpha's tf=2/length=3
    # outranks tf=1/length=4, while common's tf=3/length=4 outranks tf=1/length=3.
    assert index.search("alpha alpha") == ("tf_heavy", "length_heavy")
    assert index.search("common") == ("length_heavy", "tf_heavy")


def test_positive_idf_gives_a_rare_term_more_weight_than_a_frequent_term() -> None:
    index = ToolSearchIndex(
        (
            _document("common_heavy", "common common common common common", 0),
            _document("rare_only", "rare", 1),
            _document("common_short_1", "common", 2),
            _document("common_short_2", "common", 3),
        )
    )

    # With N=4 and average length 2, positive IDF is about 1.204 for rare and
    # 0.357 for common. That makes rare_only outrank the higher-TF common document.
    assert index.search("common rare") == (
        "rare_only",
        "common_heavy",
        "common_short_1",
    )


def test_catalog_order_breaks_equal_scores() -> None:
    index = ToolSearchIndex(
        (
            _document("later", "same", 9),
            _document("earlier", "same", 2),
        )
    )

    assert index.search("same") == ("earlier", "later")


def test_exclusions_are_applied_before_the_top_three_cap() -> None:
    index = ToolSearchIndex(
        tuple(_document(f"tool_{number}", "shared capability", number) for number in range(5))
    )

    assert index.search("shared capability") == ("tool_0", "tool_1", "tool_2")
    assert index.search("shared capability", excluded_names={"tool_0", "tool_1"}) == (
        "tool_2",
        "tool_3",
        "tool_4",
    )
    assert (
        index.search("shared capability", excluded_names={f"tool_{number}" for number in range(5)})
        == ()
    )


def test_repeated_searches_keep_fixed_statistics_and_progress_to_next_names() -> None:
    index = ToolSearchIndex(
        tuple(_document(f"tool_{number}", "shared", number) for number in range(4))
    )
    exposed: set[str] = set()

    first = index.search("shared", excluded_names=exposed)
    exposed.update(first)
    second = index.search("shared", excluded_names=exposed)
    exposed.update(second)

    assert first == ("tool_0", "tool_1", "tool_2")
    assert second == ("tool_3",)
    assert index.search("shared", excluded_names=exposed) == ()


def test_builtin_keyword_constants_cover_only_deferred_builtin_tools() -> None:
    assert MAX_TOOL_SEARCH_RESULTS == 3
    assert set(BUILTIN_TOOL_SEARCH_KEYWORDS) == {
        "web_search",
        "web_fetch",
        "schedule",
        "spawn_agent",
        "wait_agent",
        "list_agents",
    }
    assert all(keywords for keywords in BUILTIN_TOOL_SEARCH_KEYWORDS.values())
    with pytest.raises(TypeError):
        BUILTIN_TOOL_SEARCH_KEYWORDS["web_search"] = ("changed",)  # type: ignore[index]


@pytest.mark.parametrize(
    "query",
    (
        "pull request",
        "Pull Request",
        "github pull request",
        "(PULL REQUEST)",
        "拉取 pull request。",
    ),
)
def test_complete_phrase_matches_at_ascii_word_boundaries(query: str) -> None:
    index = ToolSearchIndex((ToolSearchDocument("opaque", ("Pull Request",), 0),))

    assert index.search(query) == ("opaque",)


@pytest.mark.parametrize(
    "query",
    (
        "request",
        "pull",
        "request pull",
        "pull requests",
        "pull requested",
        "get_pull request",
        "pull request_id",
        "repull request",
        "pull request2",
        "pull-request",
        "pull_request",
        "pull  request",
        "pull\trequest",
    ),
)
def test_partial_phrases_and_word_substrings_do_not_match(query: str) -> None:
    index = ToolSearchIndex((ToolSearchDocument("opaque", ("pull request",), 0),))

    assert index.search(query) == ()


@pytest.mark.parametrize("term", ("c++", "a.b", "lookup(foo)", "[entry]", "a|b"))
def test_keywords_are_literals_and_are_not_interpreted_as_regex(term: str) -> None:
    index = ToolSearchIndex((ToolSearchDocument("opaque", (term,), 0),))

    assert index.search(f"use {term.upper()} now") == ("opaque",)
    assert index.search("aaaa axb lookupfoo entry a b") == ()


def _mixed_catalog_documents() -> tuple[ToolSearchDocument, ...]:
    builtin_documents = tuple(
        ToolSearchDocument(name, (name, *keywords), order)
        for order, (name, keywords) in enumerate(BUILTIN_TOOL_SEARCH_KEYWORDS.items())
    )
    mcp_rows = (
        ("gh_lookup", "mcp_github_gh_lookup", ("github", "pull request", "code review")),
        ("slack_history", "mcp_slack_slack_history", ("slack", "channel history", "conversation")),
        ("fs_entries", "mcp_files_fs_entries", ("filesystem", "directory", "file listing")),
        *(
            (f"db_{i}", f"mcp_database_db_{i}", (f"database table {i}", "database", "sql"))
            for i in range(97)
        ),
    )
    return (
        *builtin_documents,
        *(
            ToolSearchDocument(model, (remote, model, *keywords), order + len(builtin_documents))
            for order, (remote, model, keywords) in enumerate(mcp_rows)
        ),
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    (
        ("internet lookup", "web_search"),
        ("fetch url page content", "web_fetch"),
        ("schedule reminder", "schedule"),
        ("delegate task to subagent", "spawn_agent"),
        ("wait for subagent results", "wait_agent"),
        ("list subagents", "list_agents"),
        ("github pull request", "mcp_github_gh_lookup"),
        ("slack channel history", "mcp_slack_slack_history"),
        ("filesystem directory", "mcp_files_fs_entries"),
        ("gh_lookup", "mcp_github_gh_lookup"),
        ("mcp_github_gh_lookup", "mcp_github_gh_lookup"),
    ),
)
def test_mixed_catalog_intent_queries_rank_target_first_ignoring_case(
    query: str,
    expected: str,
) -> None:
    documents = _mixed_catalog_documents()
    assert len(documents) == 106
    index = ToolSearchIndex(documents)

    result = index.search(query)

    assert result[0] == expected
    assert len(result) <= 3
    assert index.search(query.upper()) == index.search(query.title()) == result


@pytest.mark.parametrize(
    "query",
    ("pull", "request", "pull requests", "history", "listing", "gh", "lookup"),
)
def test_mixed_catalog_does_not_discover_mcp_tools_through_partial_expressions(query: str) -> None:
    index = ToolSearchIndex(_mixed_catalog_documents())

    assert all(not name.startswith("mcp_") for name in index.search(query))


def test_builtin_expressions_do_not_inherit_mcp_count_or_length_limits() -> None:
    terms = ("web", "fetch", "url", "http", "page", "content", "long " + "a" * 129)
    index = ToolSearchIndex((ToolSearchDocument("web_fetch", terms, 0),))

    assert len(terms) == 7
    for term in terms:
        assert index.search(term) == ("web_fetch",)


@pytest.mark.asyncio
async def test_tool_search_schema_has_only_required_english_query() -> None:
    tool = ToolSearchTool(lambda query: (query,))

    function = tool.to_schema()["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    assert set(parameters["properties"]) == {"query"}
    assert parameters["required"] == ["query"]
    assert "English" in parameters["properties"]["query"]["description"]
    assert "limit" not in parameters
    assert "mode" not in parameters


@pytest.mark.asyncio
async def test_tool_search_preparation_does_not_call_callback_and_invalid_input_is_rejected() -> (
    None
):
    calls: list[str] = []

    def search_and_activate(query: str) -> tuple[str, ...]:
        calls.append(query)
        return ()

    tool = ToolSearchTool(search_and_activate)
    facts = await tool.prepare({"query": "find files"})
    assert facts.normalized_arguments == {"query": "find files"}
    assert calls == []

    with pytest.raises(ToolError):
        await tool.prepare({"query": 7})
    assert calls == []


@pytest.mark.asyncio
async def test_tool_search_executes_one_async_callback_and_returns_only_json_names() -> None:
    calls: list[str] = []

    async def search_and_activate(query: str) -> tuple[str, ...]:
        calls.append(query)
        return ("mcp_search", "web_fetch")

    result = await ToolSearchTool(search_and_activate).execute(query="search issues")

    assert json.loads(result) == ["mcp_search", "web_fetch"]
    assert calls == ["search issues"]


@pytest.mark.asyncio
async def test_tool_search_executes_a_synchronous_callback() -> None:
    calls: list[str] = []

    def search_and_activate(query: str) -> tuple[str, ...]:
        calls.append(query)
        return ("web_search",)

    result = await ToolSearchTool(search_and_activate).execute(query="search the web")

    assert json.loads(result) == ["web_search"]
    assert calls == ["search the web"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("callback_result", "error_type"),
    [
        ("one_name", TypeError),
        (("one", "two", "three", "four"), ValueError),
        (("one", ""), TypeError),
        (("one", 2), TypeError),
    ],
)
async def test_tool_search_rejects_results_outside_the_name_array_contract(
    callback_result: object,
    error_type: type[Exception],
) -> None:
    def search_and_activate(query: str) -> Iterable[str]:
        del query
        return cast(Iterable[str], callback_result)

    with pytest.raises(error_type):
        await ToolSearchTool(search_and_activate).execute(query="find a tool")


@pytest.mark.asyncio
async def test_tool_search_propagates_callback_cancellation() -> None:
    started = asyncio.Event()

    async def search_and_activate(query: str) -> tuple[str, ...]:
        del query
        started.set()
        await asyncio.Future()
        return ()

    task = asyncio.create_task(ToolSearchTool(search_and_activate).execute(query="wait"))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
