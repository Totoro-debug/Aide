from __future__ import annotations

import httpx
import pytest

from aide.provider.http_capture import _safe_headers
from aide.trace.sanitization import (
    is_sensitive_trace_name,
    sanitize_trace_url,
    sanitize_trace_value,
)


@pytest.mark.parametrize(
    ("name", "sensitive"),
    [
        ("authorization", True),
        ("auth", True),
        ("api_key", True),
        ("x-api-key", True),
        ("access_token", True),
        ("refresh_token", True),
        ("token_id", True),
        ("signature", True),
        ("sig", True),
        ("credential", True),
        ("cookie", True),
        ("password", True),
        ("author", False),
        ("monkey", False),
        ("token_count", False),
        ("max_tokens", False),
    ],
)
def test_provider_and_tool_metadata_share_sensitive_name_rules(name: str, sensitive: bool) -> None:
    assert is_sensitive_trace_name(name) is sensitive
    expected = {} if sensitive else {name: "value"}
    assert sanitize_trace_value({name: "value"}) == expected
    assert _safe_headers(httpx.Headers({name: "value"})) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://user:password@example.test/p?api_key=secret&q=a%20b&q=a+b&empty=&flag#part",
            "https://example.test/p?q=a%20b&q=a+b&empty=&flag#part",
        ),
        ("/next?token_id=secret&q=%2f&q=中文#part", "/next?q=%2f&q=中文#part"),
        ("//user:password@example.test/p?sig=secret&q=1", "//example.test/p?q=1"),
        ("https://user:password@example.test?auth=secret", "https://example.test"),
        ("?%61pi_key=secret&max_tokens=2&token_count=1", "?max_tokens=2&token_count=1"),
    ],
)
def test_url_sanitization_preserves_public_query_segments(url: str, expected: str) -> None:
    assert sanitize_trace_url(url) == expected


def test_provider_sanitizes_each_header_url_and_preserves_repeated_headers() -> None:
    headers = httpx.Headers(
        [
            ("location", "/next?token_id=secret&q=a%20b&q=a+b#part"),
            ("content-location", "//user:password@example.test/p?auth=secret&empty=#part"),
            ("referer", "https://example.test/?sig=secret&flag#part"),
            (
                "link",
                '<https://example.test/p?api_key=secret&q=%2f#part>; rel="next", '
                '</p?token=secret&page=1#part>; rel="prev"',
            ),
            ("x-request-id", "first"),
            ("x-request-id", "second"),
        ]
    )
    assert _safe_headers(headers) == {
        "location": "/next?q=a%20b&q=a+b",
        "content-location": "//example.test/p?empty=",
        "referer": "https://example.test/?flag",
        "link": '<https://example.test/p?q=%2f>; rel="next", </p?page=1>; rel="prev"',
        "x-request-id": ["first", "second"],
    }
