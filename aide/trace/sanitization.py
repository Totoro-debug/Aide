"""Shared sanitization for sensitive values in Execution Trace metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import unquote_plus, urlsplit

_HTTP_URL_PATTERN = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)


def is_sensitive_trace_name(name: str) -> bool:
    normalized = "".join(character for character in name.casefold() if character.isalnum())
    sensitive_parts = (
        "authorization",
        "authentication",
        "authkey",
        "apikey",
        "accesskey",
        "accesskeyid",
        "privatekey",
        "secretkey",
        "signingkey",
        "subscriptionkey",
        "token",
        "secret",
        "credential",
        "credentials",
        "cookie",
        "password",
        "signature",
    )
    return (
        normalized in {"key", "sig"}
        or normalized.endswith("auth")
        or normalized.endswith(sensitive_parts)
    )


def sanitize_trace_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme and parts.scheme.casefold() not in {"http", "https"}:
        return url
    before_fragment, fragment_marker, fragment = url.partition("#")
    base, query_marker, query = before_fragment.partition("?")
    if "@" in parts.netloc:
        authority_start = base.index("://") + 3
        authority_end = base.find("/", authority_start)
        if authority_end == -1:
            authority_end = len(base)
        authority = base[authority_start:authority_end].rsplit("@", 1)[-1]
        base = base[:authority_start] + authority + base[authority_end:]
    safe_segments = [
        segment
        for segment in query.split("&")
        if not is_sensitive_trace_name(unquote_plus(segment.partition("=")[0]))
    ]
    safe_query = query_marker + "&".join(safe_segments) if safe_segments else ""
    return base + safe_query + fragment_marker + fragment


def sanitize_trace_text_urls(text: str) -> str:
    def replace_url(match: re.Match[str]) -> str:
        try:
            return sanitize_trace_url(match.group())
        except ValueError:
            return match.group()

    return _HTTP_URL_PATTERN.sub(replace_url, text)


def sanitize_trace_value(value: object) -> object:
    if isinstance(value, str):
        return sanitize_trace_text_urls(value)
    if isinstance(value, Mapping):
        return {
            key: sanitize_trace_value(item)
            for key, item in value.items()
            if not isinstance(key, str) or not is_sensitive_trace_name(key)
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_trace_value(item) for item in value]
    return value


__all__ = [
    "is_sensitive_trace_name",
    "sanitize_trace_text_urls",
    "sanitize_trace_url",
    "sanitize_trace_value",
]
