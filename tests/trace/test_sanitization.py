from __future__ import annotations

import pytest

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
