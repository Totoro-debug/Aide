"""Pure local retrieval primitives for deferred Tool discovery."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from math import log
from types import MappingProxyType
from typing import Final

BM25_K1: Final[float] = 1.2
BM25_B: Final[float] = 0.75
MAX_TOOL_SEARCH_RESULTS: Final[int] = 3

BUILTIN_TOOL_SEARCH_KEYWORDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "web_search": ("web", "search", "internet", "lookup"),
        "web_fetch": ("web", "fetch", "url", "http", "page", "content"),
        "schedule": ("schedule", "job", "reminder", "cron", "timer"),
    }
)

_ASCII_TOKEN = re.compile(r"[A-Za-z0-9]+")
_UPPERCASE_WORD_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
_LOWERCASE_WORD_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")


def tokenize_tool_search_text(text: str) -> tuple[str, ...]:
    """Split English search text into lowercase ASCII alphanumeric terms."""
    if not isinstance(text, str):
        raise TypeError("Tool Search text must be a string")

    separated = _UPPERCASE_WORD_BOUNDARY.sub(r"\1 \2", text)
    separated = _LOWERCASE_WORD_BOUNDARY.sub(r"\1 \2", separated)
    return tuple(token.lower() for token in _ASCII_TOKEN.findall(separated))


@dataclass(frozen=True, slots=True)
class ToolSearchDocument:
    """One callable model name and its fixed search terms in Catalog order."""

    name: str
    terms: tuple[str, ...]
    catalog_order: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("Tool Search document name must be a non-empty string")
        if isinstance(self.terms, (str, bytes)):
            raise TypeError("Tool Search document terms must be an iterable of strings")
        if not isinstance(self.catalog_order, int) or isinstance(self.catalog_order, bool):
            raise TypeError("Tool Search document catalog order must be an integer")
        if self.catalog_order < 0:
            raise ValueError("Tool Search document catalog order must be nonnegative")

        normalized_terms: list[str] = []
        for term in self.terms:
            if not isinstance(term, str):
                raise TypeError("Tool Search document terms must contain only strings")
            normalized_terms.extend(tokenize_tool_search_text(term))
        normalized = tuple(normalized_terms)
        object.__setattr__(self, "terms", normalized)


@dataclass(frozen=True, slots=True)
class _Posting:
    document_index: int
    term_frequency: int


class ToolSearchIndex:
    """Read-only inverted BM25 index over a fixed Tool document collection."""

    def __init__(self, documents: Iterable[ToolSearchDocument]) -> None:
        document_tuple = tuple(documents)
        if any(not isinstance(document, ToolSearchDocument) for document in document_tuple):
            raise TypeError("Tool Search index documents must be ToolSearchDocument instances")
        if len({document.name for document in document_tuple}) != len(document_tuple):
            raise ValueError("Tool Search document names must be unique")

        self._documents = document_tuple
        self._average_document_length = (
            sum(len(document.terms) for document in document_tuple) / len(document_tuple)
            if document_tuple
            else 0.0
        )
        postings: dict[str, list[_Posting]] = {}
        document_frequency: Counter[str] = Counter()
        for document_index, document in enumerate(document_tuple):
            term_frequencies = Counter(document.terms)
            for term, frequency in term_frequencies.items():
                postings.setdefault(term, []).append(
                    _Posting(document_index=document_index, term_frequency=frequency)
                )
                document_frequency[term] += 1

        document_count = len(document_tuple)
        self._postings: Mapping[str, tuple[_Posting, ...]] = MappingProxyType(
            {term: tuple(entries) for term, entries in postings.items()}
        )
        self._inverse_document_frequency: Mapping[str, float] = MappingProxyType(
            {
                term: log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
                for term, frequency in document_frequency.items()
            }
        )

    def search(
        self,
        query: str,
        excluded_names: Collection[str] = (),
    ) -> tuple[str, ...]:
        """Return up to three eligible model names ranked by BM25 score."""
        query_terms = tuple(dict.fromkeys(tokenize_tool_search_text(query)))
        if not query_terms:
            return ()

        excluded = frozenset(excluded_names)
        matched_frequencies: dict[int, dict[str, int]] = {}
        for term in query_terms:
            for posting in self._postings.get(term, ()):
                document = self._documents[posting.document_index]
                if document.name in excluded:
                    continue
                matched_frequencies.setdefault(posting.document_index, {})[term] = (
                    posting.term_frequency
                )

        scored: list[tuple[float, int, int]] = []
        for document_index, term_frequencies in matched_frequencies.items():
            document = self._documents[document_index]
            score = self._score(document, query_terms, term_frequencies)
            scored.append((score, document.catalog_order, document_index))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return tuple(
            self._documents[document_index].name
            for _, _, document_index in scored[:MAX_TOOL_SEARCH_RESULTS]
        )

    def _score(
        self,
        document: ToolSearchDocument,
        query_terms: tuple[str, ...],
        term_frequencies: Mapping[str, int],
    ) -> float:
        document_length = len(document.terms)
        denominator_length = (
            0.0
            if self._average_document_length == 0
            else BM25_B * document_length / self._average_document_length
        )
        normalization = BM25_K1 * (1 - BM25_B + denominator_length)
        return sum(
            self._inverse_document_frequency[term]
            * (frequency * (BM25_K1 + 1) / (frequency + normalization))
            for term in query_terms
            if (frequency := term_frequencies.get(term, 0))
        )
