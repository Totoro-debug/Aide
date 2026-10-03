"""Shared canonical text normalization rules."""

from __future__ import annotations

_TITLE_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("\u201c", "\u201d"),
    ("\u2018", "\u2019"),
    ("\u300c", "\u300d"),
    ("\u300e", "\u300f"),
    ("\u00ab", "\u00bb"),
)


def normalize_title(value: str, *, fallback: str = "Untitled session") -> str:
    """Normalize the first non-empty title line using the Session rule."""
    for line in value.splitlines():
        title = " ".join(line.split())
        if not title:
            continue
        for opening, closing in _TITLE_PAIRS:
            if len(title) >= 2 and title.startswith(opening) and title.endswith(closing):
                title = " ".join(title[1:-1].split())
                break
        return title[:60] or fallback
    return fallback


def normalize_title_candidate(value: str) -> str:
    """Normalize a title without substituting the Session fallback."""
    return normalize_title(value, fallback="")
